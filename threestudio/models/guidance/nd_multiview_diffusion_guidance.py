import cv2
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List

import threestudio
from extern.nd_sd.model_zoo import build_model
from extern.nd_sd.ldm.camera_utils import (convert_opengl_to_blender,
                                           normalize_camera,)
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *


@threestudio.register("nd-multiview-diffusion-guidance")
class MultiviewDiffusionGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        model_name: str = "nd-4view"  # check mvdream.model_zoo.PRETRAINED_MODELS
        ckpt_path: Optional[
            str
        ] = None  # path to local checkpoint (None for loading from url)
        guidance_scale: float = 50.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        camera_condition_type: str = "rotation"
        view_dependent_prompting: bool = False

        n_view: int = 4
        image_size: int = 256
        recon_loss: bool = True
        recon_std_rescale: float = 0.5
        collect_inputs: Optional[list] = field(default_factory=lambda: ["comp_rgb"])
        collect_inputs_lat: Optional[list] = field(default_factory=lambda: ["comp_rgb"])
        camera_distance: float = 2.0
        rotate_z: bool = False
        weighting_strategy: str = "sds"
        cam_method: str = "abs_spec"
        generate_img: bool = False
        half_precision_weights: bool = True

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Multiview Diffusion ...")

        self.model, self.model_cfg = build_model(
            self.cfg.model_name, ckpt_path=self.cfg.ckpt_path, return_cfg=True
        )
        self.cond_method = (
            self.model_cfg.model.params.cond_method
            if hasattr(self.model_cfg.model.params, "cond_method")
            else "ori"
        )

        from extern.nd_sd.ldm.models.diffusion.ddim import DDIMSampler

        self.sampler = DDIMSampler(self.model)

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.num_train_timesteps = 1000
        min_step_percent = C(self.cfg.min_step_percent, 0, 0)
        max_step_percent = C(self.cfg.max_step_percent, 0, 0)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
        self.grad_clip_val: Optional[float] = None

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )
                
        self.model.eval()
        self.model.to(self.device, dtype=self.weights_dtype)
        self.alphas_cumprod: Float[Tensor, "..."] = self.model.alphas_cumprod.to(self.device)

        threestudio.info(f"Loaded Multiview Diffusion!")
        self.count = 0

    def get_cond_input(self, input, cond_method: str, c: dict, image_size):
        if cond_method == "ori":
            pass
        elif cond_method == "cat_n":
            normal = input["normal"]
            normal_z = torch.nn.functional.interpolate(
                normal, size=(image_size // 8, image_size // 8), mode="nearest"
            )

            c["c_concat"] = normal_z.repeat(2, 1, 1, 1).detach()

        elif cond_method == "cat_d":
            depth = input["depth"]
            depth_z = torch.nn.functional.interpolate(
                depth, size=(image_size // 8, image_size // 8), mode="nearest"
            )
            c["c_concat"] = depth_z.repeat(2, 1, 1, 1).detach()

        elif cond_method == "cat_nd":
            normal = input["normal"]
            depth = input["depth"]

            normal_z = torch.nn.functional.interpolate(
                normal, size=(image_size // 8, image_size // 8), mode="nearest"
            )

            depth_z = torch.nn.functional.interpolate(
                depth, size=(image_size // 8, image_size // 8), mode="nearest"
            )

            nd_z = torch.cat([normal_z, depth_z], dim=1)
            c["c_concat"] = nd_z.repeat(2, 1, 1, 1).detach()

        else:
            raise NotImplementedError
        return c

    def generate_img(
        self, c, image_size, batch_size, other_inp=None, scale=10, as_latents=False
    ):
        c_ = {}
        uc_ = {}
        for k, v in c.items():
            print(k, type(v))
            if isinstance(v, torch.Tensor):
                c_[k] = v[:batch_size]
                uc_[k] = v[batch_size:]
            else:
                c_[k] = v
                uc_[k] = v

        self.model.device = self.device
        shape = [4, image_size // 8, image_size // 8]
        step = 50
        ddim_eta = 0.0

        samples_ddim, _ = self.sampler.sample(
            S=step,
            conditioning=c_,
            batch_size=batch_size,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_,
            eta=ddim_eta,
            x_T=None,
        )

        if not as_latents:
            x_sample = self.model.decode_first_stage(samples_ddim)
        else:
            x_sample = F.interpolate(
                samples_ddim, (image_size, image_size), mode="bilinear"
            )

        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255.0 * x_sample.permute(0, 2, 3, 1).cpu().numpy()

        if "depth" in other_inp:
            depth = other_inp["depth"][:, 0].cpu().numpy()
            depth = (depth + 1.0) / 2 * 255
            os.makedirs("debug", exist_ok=True)
            gen_img = np.concatenate(list(x_sample.astype(np.uint8)), axis=1)[
                :, :, (2, 1, 0)
            ]
            depth = np.concatenate(list(depth.astype(np.uint8)), axis=1)[
                :, :, np.newaxis
            ]
            depth = cv2.resize(
                np.tile(depth, (1, 1, 3)),
                dsize=(gen_img.shape[1], gen_img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            vis_img = np.concatenate([depth, gen_img], axis=0)
            cv2.imwrite(f"debug/debug_sample_{self.count}.jpg", vis_img)
        else:
            os.makedirs("debug", exist_ok=True)
            gen_img = np.concatenate(list(x_sample.astype(np.uint8)), axis=1)[
                :, :, (2, 1, 0)
            ]
            cv2.imwrite(f"debug/debug_sample_{self.count}.jpg", gen_img)
        self.count += 1

    def get_camera_cond(
        self,
        camera: Float[Tensor, "B 4 4"],
        fovy=None,
        distances=1,
    ):
        # Note: the input of threestudio is already blender coordinate system
        # camera = convert_opengl_to_blender(camera)
        if self.cfg.camera_condition_type == "rotation":  # normalized camera
            camera = normalize_camera(camera)
            camera = camera.view(-1, 4, 4)

            if self.cfg.rotate_z:
                from scipy.spatial.transform import Rotation as R

                r = R.from_euler("z", -90, degrees=True).as_matrix()
                rotate_mat = torch.eye(4, dtype=camera.dtype, device=camera.device)
                rotate_mat[:3, :3] = torch.from_numpy(r)
                rotate_mat = rotate_mat.unsqueeze(0).repeat(camera.shape[0], 1, 1)
                camera = torch.matmul(rotate_mat, camera)

            if isinstance(distances, torch.Tensor):
                distances = distances.unsqueeze(1)

            camera[..., :3, 3] = camera[..., :3, 3] * distances
            camera = camera.flatten(start_dim=1)

        else:
            raise NotImplementedError(
                f"Unknown camera_condition_type={self.cfg.camera_condition_type}"
            )
        return camera

    def encode_images(
        self, imgs: Float[Tensor, "B 3 256 256"]
    ) -> Float[Tensor, "B 4 32 32"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        latents = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(imgs.to(self.weights_dtype))
        )
        return latents.to(input_dtype)  # [B, 4, 32, 32] Latent space image

    def parse_input(self, input, cond_method):
        other_inp = {}
        if cond_method == "ori":
            raw_input = input

        elif cond_method == "cat_n":
            assert input.shape[1] == 6
            raw_input = input[:, :3]
            other_inp["normal"] = input[:, 3:] * 2 - 1.0

        elif cond_method == "cat_d":
            assert input.shape[1] == 4
            raw_input = input[:, :3]
            other_inp["depth"] = input[:, 3:] * 2 - 1.0

        elif cond_method == "cat_nd":
            assert input.shape[1] == 7
            raw_input = input[:, :3]
            other_inp["normal"] = input[:, 3:6] * 2 - 1.0
            other_inp["depth"] = input[:, 6:] * 2 - 1.0
        else:
            raise NotImplementedError
        return raw_input, other_inp

    def forward(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        camera_distances_relative: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents: bool = False,
        fovy=None,
        timestep=None,
        text_embeddings=None,
        input_is_latent=False,
        current_step_ratio=0,
        **kwargs,
    ):
        batch_size = rgb.shape[0]
        camera = c2w
        input_dtype = rgb.dtype

        rgb_BCHW = rgb.permute(0, 3, 1, 2)

        rgb_BCHW, other_inp = self.parse_input(rgb_BCHW, self.cond_method)

        if text_embeddings is None:
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )

        if input_is_latent:
            latents = rgb
        else:
            latents: Float[Tensor, "B 4 32 32"]
            if rgb_as_latents:
                # latents = F.interpolate(rgb_BCHW, (32, 32), mode='bilinear', align_corners=False)  # need [-1, 1]
                latents = F.adaptive_avg_pool2d(rgb_BCHW, (32, 32))
            else:
                # interp to 256x256 to be fed into vae.
                pred_rgb = F.interpolate(
                    rgb_BCHW,
                    (self.cfg.image_size, self.cfg.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
                # encode image into latents with vae, requires grad!
                latents = self.encode_images(pred_rgb)

        # sample timestep
        if timestep is None:
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [1],
                dtype=torch.long,
                device=latents.device,
            )
            t_expand = t.repeat(text_embeddings.shape[0])

        else:
            # assert timestep >= 0 and timestep < self.num_train_timesteps
            # t = torch.full([1], timestep, dtype=torch.long, device=latents.device)
            # t_expand = t.repeat(text_embeddings.shape[0])]
            t_expand = timestep

        # predict the noise residual with unet, NO grad!
        with torch.no_grad(), torch.autocast(dtype=self.weights_dtype, device_type="cuda"):
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.model.q_sample(latents, t, noise)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            # save input tensors for UNet
            if camera is not None:
                if self.cfg.cam_method == "abs_spec":
                    camera = self.get_camera_cond(
                        camera, fovy, distances=self.cfg.camera_distance
                    )
                elif self.cfg.cam_method == "abs":
                    camera = self.get_camera_cond(
                        camera, fovy, distances=camera_distances
                    )
                elif self.cfg.cam_method == "rel_x2":
                    camera = self.get_camera_cond(
                        camera, fovy, distances=camera_distances_relative * 2
                    )
                elif self.cfg.cam_method == "rel_xauto":
                    cam_dist = (camera_distances_relative - 0.8) / (1.0 - 0.8) * (
                        2.0 - 1.4
                    ) + 1.4
                    camera = self.get_camera_cond(camera, fovy, distances=cam_dist)
                else:
                    raise NotImplementedError

                camera = camera.repeat(2, 1).to(text_embeddings)
                context = {
                    "context": text_embeddings.to(self.weights_dtype),
                    "camera": camera.to(self.weights_dtype),
                    "num_frames": self.cfg.n_view,
                }
            else:
                context = {"context": text_embeddings.to(self.weights_dtype)}

            context = self.get_cond_input(
                input=other_inp,
                cond_method=self.cond_method,
                c=context,
                image_size=self.cfg.image_size,
            )

            if self.cfg.generate_img:
                self.generate_img(
                    context, self.cfg.image_size, batch_size, other_inp, scale=10
                )
            noise_pred = self.model.apply_model(latent_model_input, t_expand, context)

            noise_pred = noise_pred.to(input_dtype)

        # perform guidance
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(
            2
        )  # Note: flipped compared to stable-dreamfusion
        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        if self.cfg.recon_loss:
            # reconstruct x0
            latents_recon = self.model.predict_start_from_noise(
                latents_noisy, t, noise_pred
            )

            # clip or rescale x0
            if self.cfg.recon_std_rescale > 0:
                latents_recon_nocfg = self.model.predict_start_from_noise(
                    latents_noisy, t, noise_pred_text
                )
                latents_recon_nocfg_reshape = latents_recon_nocfg.view(
                    -1, self.cfg.n_view, *latents_recon_nocfg.shape[1:]
                )
                latents_recon_reshape = latents_recon.view(
                    -1, self.cfg.n_view, *latents_recon.shape[1:]
                )
                factor = (
                    latents_recon_nocfg_reshape.std([1, 2, 3, 4], keepdim=True) + 1e-8
                ) / (latents_recon_reshape.std([1, 2, 3, 4], keepdim=True) + 1e-8)

                latents_recon_adjust = latents_recon.clone() * factor.squeeze(
                    1
                ).repeat_interleave(self.cfg.n_view, dim=0)
                latents_recon = (
                    self.cfg.recon_std_rescale * latents_recon_adjust
                    + (1 - self.cfg.recon_std_rescale) * latents_recon
                )

            # x0-reconstruction loss from Sec 3.2 and Appendix
            loss = (
                0.5
                * F.mse_loss(latents, latents_recon.detach(), reduction="sum")
                / latents.shape[0]
            )
            grad = torch.autograd.grad(loss, latents, retain_graph=True)[0]

        else:
            # # Original SDS
            # # w(t), sigma_t^2
            # w = (1 - self.alphas_cumprod[t])

            if self.cfg.weighting_strategy == "sds":
                # w(t), sigma_t^2, alphas t:[0, 1000] -> [1, 0]
                w = (1 - self.alphas_cumprod[t]).view(-1, 1, 1, 1)
            elif self.cfg.weighting_strategy == "uniform":
                w = 1
            elif self.cfg.weighting_strategy == "fantasia3d":
                w = (self.alphas_cumprod[t] ** 0.5 * (1 - self.alphas_cumprod[t])).view(
                    -1, 1, 1, 1
                )
            elif self.cfg.weighting_strategy == "fantasia3d_1":
                w = (1.0 / (1 - self.alphas[t])).view(-1, 1, 1, 1)
            elif self.cfg.weighting_strategy == "fantasia3d_2":
                if current_step_ratio <= 0.2:
                    w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
                else:
                    w = (1.0 / (1 - self.alphas[t])).view(-1, 1, 1, 1)
            else:
                raise ValueError(
                    f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
                )

            grad = w * (noise_pred - noise)

            # clip grad for stable training?
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
            grad = torch.nan_to_num(grad)

            target = (latents - grad).detach()
            # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
            loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / latents.shape[0]

        return {
            "loss_sds": loss,
            "grad_norm": grad.norm(),
            "timestep": t_expand,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        min_step_percent = C(self.cfg.min_step_percent, epoch, global_step)
        max_step_percent = C(self.cfg.max_step_percent, epoch, global_step)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
