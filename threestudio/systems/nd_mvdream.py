import os
import torch
from dataclasses import dataclass, field

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *


@threestudio.register("nd-mvdream-system")
class NDMVDreamSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        visualize_samples: bool = False
        latent_steps: int = 1000
        nd_latent_steps: int = 1000
        texture: bool = True
        do_init: bool = False

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.has_nd_guidanece = (self.cfg.nd_guidance_type != "none") and hasattr(
            self.cfg.loss, "lambda_nd"
        )  # and (self.cfg.loss.lambda_nd > 0)
        self.has_rgb_sd_guidanece = (self.cfg.guidance_type != "none") and hasattr(
            self.cfg.loss, "lambda_rgb_sd"
        )  # and (self.cfg.loss.lambda_rgb_sd > 0)
        threestudio.info(
            f"================has_nd_guidanece:{self.has_nd_guidanece}, has_rgb_sd_guidanece:{self.has_rgb_sd_guidanece}================="
        )

        if self.has_rgb_sd_guidanece:
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
            # self.guidance.requires_grad_(False)
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )
            self.prompt_utils = self.prompt_processor()

        if self.has_nd_guidanece:
            self.nd_guidance = threestudio.find(self.cfg.nd_guidance_type)(
                self.cfg.nd_guidance
            )
            self.nd_guidance.requires_grad_(False)
            self.nd_prompt_processor = threestudio.find(
                self.cfg.nd_prompt_processor_type
            )(self.cfg.nd_prompt_processor)
            self.nd_prompt_utils = self.nd_prompt_processor()

    def on_load_checkpoint(self, checkpoint):
        for k in list(checkpoint["state_dict"].keys()):
            if k.startswith("guidance."):
                return
            if k.startswith("nd_guidance."):
                return
        if self.has_rgb_sd_guidanece:
            if hasattr(self.guidance, "state_dict"):
                guidance_state_dict = {
                    "guidance." + k: v for (k, v) in self.guidance.state_dict().items()
                }
                checkpoint["state_dict"] = {
                    **checkpoint["state_dict"],
                    **guidance_state_dict,
                }

        if self.has_nd_guidanece:
            guidance_nd_state_dict = {
                "nd_guidance." + k: v
                for (k, v) in self.nd_guidance.state_dict().items()
            }
            checkpoint["state_dict"] = {
                **checkpoint["state_dict"],
                **guidance_nd_state_dict,
            }

        return

    def on_save_checkpoint(self, checkpoint):
        for k in list(checkpoint["state_dict"].keys()):
            if k.startswith("guidance."):
                checkpoint["state_dict"].pop(k)
            if k.startswith("nd_guidance."):
                checkpoint["state_dict"].pop(k)
        return

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.renderer(
            **batch, render_rgb=self.cfg.texture or self.has_rgb_sd_guidanece
        )

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # # only used in training
        # self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
        #     self.cfg.prompt_processor
        # )
        # self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        if not self.cfg.texture or self.cfg.do_init:
            # initialize SDF
            # FIXME: what if using other geometry types?
            self.geometry.initialize_shape()

    def collect_inputs(self, out, collect_inputs):
        inputs = [out[key] for key in collect_inputs]
        return torch.cat(inputs, dim=-1)

    def training_step(self, batch, batch_idx):
        out = self(batch)

        loss = 0.0

        self.has_nd_guidanece = (
            (self.cfg.nd_guidance_type != "none")
            and hasattr(self.cfg.loss, "lambda_nd")
            and (self.C(self.cfg.loss.lambda_nd) > 0)
        )
        self.has_rgb_sd_guidanece = (
            (self.cfg.guidance_type != "none")
            and hasattr(self.cfg.loss, "lambda_rgb_sd")
            and (self.C(self.cfg.loss.lambda_rgb_sd) > 0)
        )
        print(
            f"has_nd_guidanece:{self.has_nd_guidanece}, has_rgb_sd_guidanece:{self.has_rgb_sd_guidanece}"
        )

        if not self.cfg.texture:  # geometry training
            if self.has_nd_guidanece:
                if self.true_global_step < self.cfg.nd_latent_steps:
                    nd_guidance_inp = self.collect_inputs(
                        out, self.cfg.nd_guidance.collect_inputs_lat
                    )
                    nd_guidance_inp = nd_guidance_inp * 2.0 - 1.0

                    nd_guidance_out = self.nd_guidance(
                        nd_guidance_inp,
                        self.nd_prompt_utils,
                        **batch,
                        rgb_as_latents=True,
                    )
                else:
                    nd_guidance_inp = self.collect_inputs(
                        out, self.cfg.nd_guidance.collect_inputs
                    )

                    nd_guidance_out = self.nd_guidance(
                        nd_guidance_inp,
                        self.nd_prompt_utils,
                        **batch,
                        rgb_as_latents=False,
                    )

            if self.has_rgb_sd_guidanece:
                timestep = (
                    nd_guidance_out["timestep"]
                    if self.cfg.guidance.share_t and self.has_nd_guidanece
                    else None
                )
                if self.true_global_step < self.cfg.latent_steps:
                    guidance_inp = self.collect_inputs(
                        out, self.cfg.guidance.collect_inputs_lat
                    )
                    guidance_inp = guidance_inp * 2.0 - 1.0

                    guidance_out = self.guidance(
                        guidance_inp,
                        self.prompt_utils,
                        **batch,
                        rgb_as_latents=True,
                        timestep=timestep,
                    )
                else:
                    collect_inps = self.cfg.guidance.collect_inputs
                    if self.cfg.switch_ginp:
                        collect_inps = [
                            collect_inps[
                                self.true_global_step % self.cfg.switch_freq == 0
                            ]
                        ]
                        print(self.true_global_step, collect_inps)

                    guidance_inp = self.collect_inputs(out, collect_inps)

                    guidance_out = self.guidance(
                        guidance_inp,
                        self.prompt_utils,
                        **batch,
                        rgb_as_latents=False,
                        timestep=timestep,
                    )

            if "mesh" in out:
                if (
                    hasattr(self.cfg.loss, "lambda_normal_consistency")
                    and self.C(self.cfg.loss.lambda_normal_consistency) > 0
                ):
                    loss_normal_consistency = out["mesh"].normal_consistency()
                    self.log("train/loss_normal_consistency", loss_normal_consistency)
                    loss += loss_normal_consistency * self.C(
                        self.cfg.loss.lambda_normal_consistency
                    )
                if (
                    hasattr(self.cfg.loss, "lambda_laplacian_smoothness")
                    and self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0
                ):
                    loss_laplacian_smoothness = out["mesh"].laplacian()
                    self.log(
                        "train/loss_laplacian_smoothness", loss_laplacian_smoothness
                    )
                    loss += loss_laplacian_smoothness * self.C(
                        self.cfg.loss.lambda_laplacian_smoothness
                    )
        else:  # texture training
            if self.has_nd_guidanece:
                nd_guidance_inp = self.collect_inputs(
                    out, self.cfg.nd_guidance.collect_inputs
                )
                nd_guidance_out = self.nd_guidance(
                    nd_guidance_inp, self.nd_prompt_utils, **batch, rgb_as_latents=False
                )

            if self.has_rgb_sd_guidanece:
                timestep = (
                    nd_guidance_out["timestep"]
                    if self.cfg.guidance.share_t and self.has_nd_guidanece
                    else None
                )
                guidance_inp = self.collect_inputs(
                    out, self.cfg.guidance.collect_inputs
                )
                guidance_out = self.guidance(
                    guidance_inp,
                    self.prompt_utils,
                    **batch,
                    rgb_as_latents=False,
                    timestep=timestep,
                    current_step_ratio=self.true_global_step / self.trainer.max_steps,
                )

        loss_rgb_sd = 0
        if self.has_rgb_sd_guidanece:
            for name, value in guidance_out.items():
                if name != "timestep":
                    self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss_rgb_sd += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_rgb_")]
                    )
            loss += loss_rgb_sd  # * self.C(self.cfg.loss.lambda_rgb_sd)

        loss_nd = 0
        if self.has_nd_guidanece:
            for name, value in nd_guidance_out.items():
                if name != "timestep":
                    self.log(f"train/nd_{name}", value)
                if name.startswith("loss_"):
                    loss_nd += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_nd_")]
                    )
            nd_wight = (
                self.C(self.cfg.loss.lambda_nd_w)
                if hasattr(self.cfg.loss, "lambda_nd_w")
                else 1.0
            )
            loss += loss_nd * nd_wight  # * self.C(self.cfg.loss.lambda_nd)
        print(f"loss_rgb_sd:{loss_rgb_sd}, loss_nd:{loss_nd}")

        # if self.cfg.texture:
        #     for name, value in self.cfg.loss.items():
        #         self.log(f"train_params/{name}", self.C(value))
        #     return {"loss": loss}

        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                out["weights"].detach()
                * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (out["opacity"] > 0).sum()
            self.log("train/loss_orient", loss_orient)
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        if self.C(self.cfg.loss.lambda_sparsity) > 0:
            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        if self.C(self.cfg.loss.lambda_opaque) > 0:
            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
        # helps reduce floaters and produce solid geometry
        if self.C(self.cfg.loss.lambda_z_variance) > 0:
            loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
            self.log("train/loss_z_variance", loss_z_variance)
            loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

        if (
            hasattr(self.cfg.loss, "lambda_eikonal")
            and self.C(self.cfg.loss.lambda_eikonal) > 0
        ):
            loss_eikonal = (
                (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
            ).mean()
            self.log("train/loss_eikonal", loss_eikonal)
            loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step:06d}-{batch['index'][0]:04d}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            # + (
            #     [
            #         {
            #             "type": "rgb",
            #             "img": out["comp_normal_white_vis"][0],
            #             "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
            #         }
            #     ]
            #     if "comp_normal_white_vis" in out
            #     else []
            # )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal_cam_white_vis"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal_cam_white_vis" in out
                else []
            ),
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step:06d}-test/{batch['index'][0]:04d}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            # + (
            #     [
            #         {
            #             "type": "rgb",
            #             "img": out["comp_normal_white_vis"][0],
            #             "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
            #         }
            #     ]
            #     if "comp_normal_white_vis" in out
            #     else []
            # )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal_cam_white_vis"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal_cam_white_vis" in out
                else []
            ),
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
                f"it{self.true_global_step:06d}-test",
                f"it{self.true_global_step:06d}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
