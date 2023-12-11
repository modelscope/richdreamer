import nerfacc
import pdb
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from einops import repeat

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import Rasterizer, VolumeRenderer
from threestudio.utils.misc import get_device
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.transfer import rotate_y
from threestudio.utils.typing import *


def c2wtow2c(c2w):
    """transfer camera2world to world2camera matrix"""

    w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
    w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
    w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
    w2c[:, 3, 3] = 1.0

    return w2c


def homo_proj(p: Float[Tensor, "B n 3"], matrix: Float[Tensor, "B 4 4"]):
    # NOTE that only at homo_weight = 1

    r_p: Float[Tensor, "B n 3"] = (matrix[:, :3, :3] @ p.transpose(2, 1)).transpose(
        2, 1
    )
    t_p: Float[Tensor, "B 1 3"] = matrix[:, :3, 3][:, None, :]
    return r_p + t_p


class _Base_NVDiffRasterizerDMTet(Rasterizer):
    """
    NOTE that the normalized value by disparity has scale issue,
    we need to time scale factor

    """

    @dataclass
    class Config(VolumeRenderer.Config):
        context_type: str = "gl"
        camera_space: bool = False
        near_distance: float = 1.732

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)

        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, get_device())

    @staticmethod
    def world2camera(normal, w2c):
        rotate: Float[Tensor, "B 4 4"] = w2c[..., :3, :3]
        camera_normal = normal @ rotate.permute(0, 2, 1)
        # pixel space flip axis so we need built negative y-axis normal
        flip_x = torch.eye(3).to(w2c)
        flip_x[0, 0] = -1

        camera_normal = camera_normal @ flip_x[None, ...]

        return camera_normal

    def pixelize(
        self,
        out,
        batch_size,
        height,
        width,
        selector,
        rast,
        v_pos_clip,
        mesh,
        gb_viewdirs,
        mask,
    ):
        pixel_out = {}

        for key in out.keys():
            rgb_fg = out[key]
            N, c = rgb_fg.shape

            gb_rgb_fg = torch.zeros(batch_size, height, width, c).to(rgb_fg)
            gb_rgb_fg[selector] = rgb_fg

            gb_rgb_bg = self.background(dirs=gb_viewdirs)[..., :c]
            gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.float())
            gb_rgb_aa = self.ctx.antialias(gb_rgb, rast, v_pos_clip, mesh.t_pos_idx)

            if key == "color":
                pixel_out["comp_rgb"] = gb_rgb_aa
            else:
                pixel_out[key] = gb_rgb_aa

        return pixel_out

    def forward(
        self,
        mvp_mtx: Float[Tensor, "B 4 4"],
        camera_positions: Float[Tensor, "B 3"],
        light_positions: Float[Tensor, "B 3"],
        height: int,
        width: int,
        render_rgb: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        raise NotImplementedError


@threestudio.register("nvdiff-rasterizer-dmtet")
class NVDiffRasterizerDMTet(_Base_NVDiffRasterizerDMTet):
    """
    using different bg for sd and colors
    """

    @dataclass
    class Config(_Base_NVDiffRasterizerDMTet.Config):
        mv_bg_colors: str = "blue"
        sd_bg_colors: str = "blue"

    cfg: Config

    def obtain_bg_colors(self, gb_normal, mask, bg_colors="blue"):
        if bg_colors == "blue":
            _a = torch.zeros_like(gb_normal)
            _a[..., 2] = 1.0
            _a = (_a + 1.0) / 2.0
        elif bg_colors == "black":
            _a = torch.zeros_like(gb_normal)
        elif bg_colors == "white":
            _a = torch.ones_like(gb_normal)
        else:
            raise NotImplementedError

        _b = (gb_normal + 1.0) / 2.0
        gb_normal_aa = torch.lerp(_a, _b, mask.float())
        return gb_normal_aa

    def forward(
        self,
        mvp_mtx: Float[Tensor, "B 4 4"],
        camera_positions: Float[Tensor, "B 3"],
        light_positions: Float[Tensor, "B 3"],
        height: int,
        width: int,
        render_rgb: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        batch_size = mvp_mtx.shape[0]

        # v_pos, t_pos_id -> (v,vertices, t-> face)
        mesh = self.geometry.isosurface()

        v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
            mesh.v_pos, mvp_mtx
        )
        # (512, 512)
        # rast return is [u,v, depth, triangle-idx]
        # NOTE that triangle-idx, offset by one. zero means not a triangle covers.
        rast, _ = self.ctx.rasterize(v_pos_clip, mesh.t_pos_idx, (height, width))
        # rast is return k

        mask = rast[..., 3:] > 0
        mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, mesh.t_pos_idx)

        out = {"opacity": mask_aa, "mesh": mesh}

        w2c = c2wtow2c(kwargs["c2w"])
        batch = w2c.shape[0]

        # for normal
        if not self.cfg.camera_space:
            v_nrm = mesh.v_nrm
            gb_normal, _ = self.ctx.interpolate_one(mesh.v_nrm, rast, mesh.t_pos_idx)
            gb_normal = F.normalize(gb_normal, dim=-1)  # using to shading normal

            mv_normal_aa = self.obtain_bg_colors(gb_normal, mask, self.cfg.mv_bg_colors)
            sd_normal_aa = self.obtain_bg_colors(gb_normal, mask, self.cfg.sd_bg_colors)

            # *********************define camera normal map ********************#
            camera_batch_v_nrm = repeat(mesh.v_nrm, "n c -> b n c", b=batch)
            camera_normal = self.world2camera(camera_batch_v_nrm, w2c)
            camera_gb_normal, _ = self.ctx.interpolate(
                camera_normal, rast, mesh.t_pos_idx
            )
            camera_gb_normal = F.normalize(camera_gb_normal, dim=-1)

            camera_a = torch.zeros_like(camera_gb_normal)  # black normal background
            camera_a[..., 2] = 1.0
            camera_a = (camera_a + 1.0) / 2.0

            camera_a = torch.ones_like(camera_gb_normal)  # black normal background

            camera_b = (camera_gb_normal + 1.0) / 2.0
            camera_gb_normal_aa = torch.lerp(camera_a, camera_b, mask.float())
            camera_gb_normal_aa = self.ctx.antialias(
                camera_gb_normal_aa, rast, v_pos_clip, mesh.t_pos_idx
            )

            out.update({"camera_normal": camera_gb_normal_aa})
            # *********************define camera normal map ********************#

        else:
            batch_v_nrm = repeat(mesh.v_nrm, "n c -> b n c", b=batch)
            camera_normal = self.world2camera(batch_v_nrm, w2c)
            gb_normal, _ = self.ctx.interpolate(camera_normal, rast, mesh.t_pos_idx)
            gb_normal = F.normalize(gb_normal, dim=-1)

            # front_colors: str = "blue"
            # bg_colors: str = "blue"
            mv_normal_aa = self.obtain_bg_colors(gb_normal, mask, self.cfg.mv_bg_colors)
            sd_normal_aa = self.obtain_bg_colors(gb_normal, mask, self.cfg.sd_bg_colors)

        # NOTE that after antialias the normal map can be large than one
        mv_normal_aa = self.ctx.antialias(
            mv_normal_aa, rast, v_pos_clip, mesh.t_pos_idx
        )

        sd_normal_aa = self.ctx.antialias(
            sd_normal_aa, rast, v_pos_clip, mesh.t_pos_idx
        )

        out.update(
            {
                "comp_normal": mv_normal_aa.clamp(0, 1),
                "sd_comp_normal": sd_normal_aa.clamp(0, 1),
                "mask": mask,
            }
        )  # in [0, 1]

        # for depth
        batch_v_pos = repeat(mesh.v_pos, "n c -> b n c", b=batch)
        # homo matrix
        camera_v_pos = homo_proj(batch_v_pos, w2c)
        depth_appe = -(camera_v_pos[..., 2:] + 1e-9)
        depth, _ = self.ctx.interpolate(depth_appe, rast, mesh.t_pos_idx)

        # mv_disparity
        # 1. comnpute disparity_free scale
        near_d = kwargs["camera_distances"] - self.cfg.near_distance
        far_d = kwargs["camera_distances"] + self.cfg.near_distance

        near_d = near_d[:, None, None, None].expand_as(depth)
        far_d = far_d[:, None, None, None].expand_as(depth)
        depth[depth > far_d] = far_d[depth > far_d]

        mv_disparity = (far_d - depth) / (2 * self.cfg.near_distance)  # 2 * sqrt(3)

        mv_disparity_aa = torch.lerp(
            torch.zeros_like(mv_disparity), mv_disparity, mask.float()
        )
        # [0, 1]
        mv_disparity_aa = self.ctx.antialias(
            mv_disparity_aa, rast, v_pos_clip, mesh.t_pos_idx
        )
        out.update({"mv_disparity": mv_disparity_aa.clamp(0, 1)})  # in [0, 1]

        if render_rgb:
            selector = mask[..., 0]

            gb_pos, _ = self.ctx.interpolate_one(mesh.v_pos, rast, mesh.t_pos_idx)
            gb_viewdirs = F.normalize(
                gb_pos - camera_positions[:, None, None, :], dim=-1
            )

            gb_light_positions = light_positions[:, None, None, :].expand(
                -1, height, width, -1
            )

            positions = gb_pos[selector]
            geo_out = self.geometry(positions, output_normal=False)

            extra_geo_info = {}
            if self.material.requires_normal:
                extra_geo_info["shading_normal"] = gb_normal[selector]
            if self.material.requires_tangent:
                gb_tangent, _ = self.ctx.interpolate_one(
                    mesh.v_tng, rast, mesh.t_pos_idx
                )
                gb_tangent = F.normalize(gb_tangent, dim=-1)
                extra_geo_info["tangent"] = gb_tangent[selector]

            mat_out = self.material(
                viewdirs=gb_viewdirs[selector],
                positions=positions,
                light_positions=gb_light_positions[selector],
                **extra_geo_info,
                **geo_out
            )

            mat_out = self.pixelize(
                mat_out,
                batch_size,
                height,
                width,
                selector,
                rast,
                v_pos_clip,
                mesh,
                gb_viewdirs,
                mask,
            )
            out.update(mat_out)

        return out
