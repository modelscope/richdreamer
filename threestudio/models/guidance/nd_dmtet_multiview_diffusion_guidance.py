import numpy as np
import pdb
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List

import threestudio
from extern.nd_sd.ldm.camera_utils import (convert_opengl_to_blender,
                                           normalize_camera,)
from extern.nd_sd.model_zoo import build_model
from extern.nd_sd.ldm.models.diffusion.ddim import DDIMSampler
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseModule, BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *


@threestudio.register("nd-mv-dmtet-guidance")
class MultiviewDiffusionDMTetGuidance(
    BaseObject
):  # using object avoid to save diffusion weights
    @dataclass
    class Config(BaseObject.Config):
        model_name: str = (
            "sd-v2.1-base-4view"  # check mvdream.model_zoo.PRETRAINED_MODELS
        )
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
        weighting_strategy: str = "sds"
        rotate_z: bool = True
        interpolate_mode: str = "bilinear"

    cfg: Config

    def collect_inputs(self, out):
        inputs = [out[key] for key in self.cfg.collect_inputs]
        return torch.cat(inputs, dim=-1)

    def to(self, device):
        self.model.to(device)

    def configure(self) -> None:
        threestudio.info(f"Loading Multiview Diffusion ...")

        self.model = build_model(self.cfg.model_name, ckpt_path=self.cfg.ckpt_path)

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.num_train_timesteps = 1000
        min_step_percent = C(self.cfg.min_step_percent, 0, 0)
        max_step_percent = C(self.cfg.max_step_percent, 0, 0)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
        self.grad_clip_val: Optional[float] = None

        self.to(self.device)
        self.alphas: Float[Tensor, "..."] = self.model.alphas_cumprod.to(self.device)

        threestudio.info(f"Loaded Multiview Diffusion!")

    def get_camera_cond(
        self,
        camera: Float[Tensor, "B 4 4"],
        fovy=None,
        distances=None,
    ):
        # Note: the input of threestudio is already blender coordinate system
        # camera = convert_opengl_to_blender(camera)
        if self.cfg.camera_condition_type == "rotation":  # normalized camera
            camera = normalize_camera(camera)
            camera = camera.view(-1, 4, 4)

            # debias mv to front view from camera 0
            if self.cfg.rotate_z:
                from scipy.spatial.transform import Rotation as R

                r = R.from_euler("z", -90, degrees=True).as_matrix()
                rotate_mat = torch.eye(4, dtype=camera.dtype, device=camera.device)
                rotate_mat[:3, :3] = torch.from_numpy(r)
                rotate_mat = rotate_mat.unsqueeze(0).repeat(camera.shape[0], 1, 1)
                camera = torch.matmul(rotate_mat, camera)

            camera[..., :3, 3] = camera[..., :3, 3] * distances[:, None]
            camera = camera.flatten(start_dim=1)
        else:
            raise NotImplementedError(
                f"Unknown camera_condition_type={self.cfg.camera_condition_type}"
            )
        return camera

    def encode_images(
        self, imgs: Float[Tensor, "B 3 256 256"]
    ) -> Float[Tensor, "B 4 32 32"]:
        imgs = imgs * 2.0 - 1.0
        latents = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(imgs)
        )
        return latents  # [B, 4, 32, 32] Latent space image

    def __call__(
        self,
        out: dict,
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        relative_distances: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents: bool = False,
        fovy=None,
        timestep=None,
        text_embeddings=None,
        input_is_latent=False,
        **kwargs,
    ):
        relative_scale_distances = 2 * relative_distances

        rgb = self.collect_inputs(out)
        batch_size = rgb.shape[0]
        camera = c2w

        rgb_BCHW = rgb.permute(0, 3, 1, 2)

        if text_embeddings is None:
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )

        if input_is_latent:
            latents = rgb
        else:
            latents: Float[Tensor, "B 4 64 64"]
            if rgb_as_latents:
                # [-1,1]
                latents = F.interpolate(
                    rgb_BCHW,
                    (self.cfg.image_size // 8, self.cfg.image_size // 8),
                    mode=self.cfg.interpolate_mode,
                    align_corners=False,
                )
            else:
                # interp to 512x512 to be fed into vae.
                pred_rgb = F.interpolate(
                    rgb_BCHW,
                    (self.cfg.image_size, self.cfg.image_size),
                    mode=self.cfg.interpolate_mode,
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
        else:
            assert timestep >= 0 and timestep < self.num_train_timesteps
            t = torch.full([1], timestep, dtype=torch.long, device=latents.device)

        t_expand = t.repeat(text_embeddings.shape[0])

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.model.q_sample(latents, t, noise)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            # save input tensors for UNet
            if camera is not None:
                camera = self.get_camera_cond(
                    camera, fovy, distances=relative_scale_distances
                )
                camera = camera.repeat(2, 1).to(text_embeddings)
                context = {
                    "context": text_embeddings,
                    "camera": camera,
                    "num_frames": self.cfg.n_view,
                }
            else:
                context = {"context": text_embeddings}
            noise_pred = self.model.apply_model(latent_model_input, t_expand, context)

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
            if self.cfg.weighting_strategy == "sds":
                # w(t), sigma_t^2
                w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
            elif self.cfg.weighting_strategy == "uniform":
                w = 1
            elif self.cfg.weighting_strategy == "fantasia3d":
                w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
            elif self.cfg.weighting_strategy == "fantasia3d_1":  # following Fantasia3D
                w = (1.0 / (1 - self.alphas[t])).view(-1, 1, 1, 1)
            elif self.cfg.weighting_strategy == "fantasia3d_2":  # following Fantasia3D
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

        return {"loss_sds": loss, "grad_norm": grad.norm(), "timestep": t}

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        min_step_percent = C(self.cfg.min_step_percent, epoch, global_step)
        max_step_percent = C(self.cfg.max_step_percent, epoch, global_step)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)


@threestudio.register("nd-mv-dmtet-cat-guidance")
class MultiviewDiffusionDMTetCatGuidance(MultiviewDiffusionDMTetGuidance):
    @dataclass
    class Config(MultiviewDiffusionDMTetGuidance.Config):
        cond_method: str = ""

    def configure(self) -> None:
        super().configure()
        self.sampler = DDIMSampler(self.model)

    def get_cond_input(self, input, cond_method: str, c: dict, image_size):
        if cond_method == "ori":
            pass
        elif cond_method == "cat_n":
            normal = input["normal"]
            normal_z = torch.nn.functional.interpolate(
                normal, size=(image_size // 8, image_size // 8), mode="nearest"
            )

            c["c_concat"] = normal_z.repeat(2, 1, 1, 1)

        elif cond_method == "cat_d":
            depth = input["depth"]
            depth_z = torch.nn.functional.interpolate(
                depth, size=(image_size // 8, image_size // 8), mode="nearest"
            )
            c["c_concat"] = depth_z.repeat(2, 1, 1, 1)

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
            c["c_concat"] = nd_z.repeat(2, 1, 1, 1)

        else:
            raise NotImplementedError
        return c

    def __call__(
        self,
        out: dict,
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        relative_distances: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents: bool = False,
        fovy=None,
        timestep=None,
        text_embeddings=None,
        input_is_latent=False,
        current_step_ratio=None,
        **kwargs,
    ):
        relative_scale_distances = 2 * relative_distances
        rgb = self.collect_inputs(out)

        batch_size = rgb.shape[0]
        camera = c2w

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        if self.cfg.cond_method == "cat_d":
            rgb_BCHW, other_inp = rgb_BCHW[:, :3], rgb_BCHW[:, 3:]  # [1,1]
            other_inp = {"depth": other_inp * 2.0 - 1}
        else:
            raise NotImplementedError

        if text_embeddings is None:
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )

        if input_is_latent:
            latents = rgb
        else:
            latents: Float[Tensor, "B 4 64 64"]
            if rgb_as_latents:
                # [-1,1]
                latents = F.interpolate(
                    rgb_BCHW,
                    (self.cfg.image_size // 8, self.cfg.image_size // 8),
                    mode=self.cfg.interpolate_mode,
                    align_corners=False,
                )
            else:
                # interp to 512x512 to be fed into vae.
                pred_rgb = F.interpolate(
                    rgb_BCHW,
                    (self.cfg.image_size, self.cfg.image_size),
                    mode=self.cfg.interpolate_mode,
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
        else:
            assert timestep >= 0 and timestep < self.num_train_timesteps
            t = torch.full([1], timestep, dtype=torch.long, device=latents.device)

        t_expand = t.repeat(text_embeddings.shape[0])

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.model.q_sample(latents, t, noise)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            # save input tensors for UNet
            if camera is not None:
                camera = self.get_camera_cond(
                    camera, fovy, distances=relative_scale_distances
                )
                camera = camera.repeat(2, 1).to(text_embeddings)
                context = {
                    "context": text_embeddings,
                    "camera": camera,
                    "num_frames": self.cfg.n_view,
                }
            else:
                context = {"context": text_embeddings}

            context = self.get_cond_input(
                input=other_inp,
                cond_method=self.cfg.cond_method,
                c=context,
                image_size=self.cfg.image_size,
            )
            # self.generate_img(context, self.cfg.image_size, batch_size, other_inp, scale=10)

            noise_pred = self.model.apply_model(latent_model_input, t_expand, context)

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
            if self.cfg.weighting_strategy == "sds":
                # w(t), sigma_t^2
                w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
            elif self.cfg.weighting_strategy == "uniform":
                w = 1
            elif self.cfg.weighting_strategy == "fantasia3d":
                w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
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

        return {"loss_sds": loss, "grad_norm": grad.norm(), "timestep": t}

    def generate_img(
        self, c, image_size, batch_size, other_inp=None, scale=10, as_latents=False
    ):
        c_ = {}
        uc_ = {}
        for k, v in c.items():
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

        # depth = c_["c_concat"][:, 0].cpu().numpy()
        depth = other_inp["depth"][:, 0].cpu().numpy()
        depth = (depth + 1.0) / 2 * 255
        import cv2
        import os

        os.makedirs("debug", exist_ok=True)
        # cv2.imwrite("debug_depth.jpg",)
        gen_img = np.concatenate(list(x_sample.astype(np.uint8)), axis=1)[
            :, :, (2, 1, 0)
        ]
        depth = np.concatenate(list(depth.astype(np.uint8)), axis=1)[:, :, np.newaxis]
        depth = cv2.resize(
            np.tile(depth, (1, 1, 3)),
            dsize=(gen_img.shape[1], gen_img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        vis_img = np.concatenate([depth, gen_img], axis=0)

        # TODO save vis_img by your self
        # save_img
