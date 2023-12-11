import envlight
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field

import threestudio
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import dot, get_activation
from threestudio.utils.typing import *


@threestudio.register("no-material-shading")
class NoMaterial(BaseMaterial):
    @dataclass
    class Config(BaseMaterial.Config):
        n_output_dims: int = 3
        color_activation: str = "sigmoid"
        input_feature_dims: Optional[int] = None
        mlp_network_config: Optional[dict] = None
        requires_normal: bool = True

        ambient_light_color: Tuple[float, float, float] = (0.1, 0.1, 0.1)
        diffuse_light_color: Tuple[float, float, float] = (0.9, 0.9, 0.9)
        ambient_only_steps: int = 1000
        diffuse_prob: float = 0.75
        textureless_prob: float = 0.5
        albedo_activation: str = "sigmoid"
        soft_shading: bool = False

        material_activation: str = "sigmoid"
        environment_texture: str = "load/lights/mud_road_puresky_1k.hdr"
        environment_scale: float = 2.0
        min_metallic: float = 0.0
        max_metallic: float = 0.9
        min_roughness: float = 0.08
        max_roughness: float = 0.9
        use_bump: bool = False

        pbr: bool = False
        pl: bool = False

    cfg: Config

    def configure(self) -> None:
        self.use_network = False
        if (
            self.cfg.input_feature_dims is not None
            and self.cfg.mlp_network_config is not None
        ):
            self.network = get_mlp(
                self.cfg.input_feature_dims,
                self.cfg.n_output_dims,
                self.cfg.mlp_network_config,
            )
            self.use_network = True
        self.requires_normal = self.cfg.requires_normal

        if not self.cfg.pl:
            self.ambient_light_color = torch.as_tensor(
                self.cfg.ambient_light_color, dtype=torch.float32
            )
            self.diffuse_light_color = torch.as_tensor(
                self.cfg.diffuse_light_color, dtype=torch.float32
            )

        self.ambient_light_color_bak = torch.as_tensor(
            (0.1, 0.1, 0.1), dtype=torch.float32
        )
        self.diffuse_light_color_bak = torch.as_tensor(
            (0.9, 0.9, 0.9), dtype=torch.float32
        )

        if self.cfg.pbr:
            self.light = envlight.EnvLight(
                self.cfg.environment_texture, scale=self.cfg.environment_scale
            )

            FG_LUT = torch.from_numpy(
                np.fromfile("load/lights/bsdf_256_256.bin", dtype=np.float32).reshape(
                    1, 256, 256, 2
                )
            )
            self.register_buffer("FG_LUT", FG_LUT)

        if self.cfg.pl:
            self.register_buffer(
                "ambient_light_color",
                torch.as_tensor(self.cfg.ambient_light_color, dtype=torch.float32),
            )
            self.register_buffer(
                "diffuse_light_color",
                torch.as_tensor(self.cfg.diffuse_light_color, dtype=torch.float32),
            )

    def forward(
        self,
        features: Float[Tensor, "B ... Nf"],
        positions: Float[Tensor, "B ... 3"],
        shading_normal: Float[Tensor, "B ... 3"],
        light_positions: Float[Tensor, "B ... 3"],
        ambient_ratio: Optional[float] = None,
        shading: Optional[str] = None,
        **kwargs
    ) -> Float[Tensor, "B ... Nc"]:
        features = features[..., :3]

        color = get_activation(self.cfg.color_activation)(features)
        color = torch.ones_like(color) * 0.7

        # diffuse_light_color = self.diffuse_light_color.to(color.device)
        # ambient_light_color = self.ambient_light_color.to(color.device)
        diffuse_light_color = self.diffuse_light_color_bak.to(color.device)
        ambient_light_color = self.ambient_light_color_bak.to(color.device)
        print(diffuse_light_color)

        light_directions: Float[Tensor, "B ... 3"] = F.normalize(
            light_positions - positions, dim=-1
        )
        diffuse_light: Float[Tensor, "B ... 3"] = (
            dot(shading_normal, light_directions).clamp(min=0.0) * diffuse_light_color
        )
        textureless_color = diffuse_light + ambient_light_color
        # clamp albedo to [0, 1] to compute shading
        color = color.clamp(0.0, 1.0) * textureless_color

        return color

    def export(self, features: Float[Tensor, "*N Nf"], **kwargs) -> Dict[str, Any]:
        color = self(features, **kwargs).clamp(0, 1)
        assert color.shape[-1] >= 3, "Output color must have at least 3 channels"
        if color.shape[-1] > 3:
            threestudio.warn(
                "Output color has >3 channels, treating the first 3 as RGB"
            )
        return {"albedo": color[..., :3]}
