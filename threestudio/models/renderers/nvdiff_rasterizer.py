import math
import nerfacc
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


@threestudio.register("nvdiff-rasterizer")
class NVDiffRasterizer(Rasterizer):
    @dataclass
    class Config(VolumeRenderer.Config):
        context_type: str = "gl"
        depth_norm_radius: float = 1.0

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        self.ctx = NVDiffRasterizerContext(self.cfg.context_type, get_device())

    def convert_pose(self, C2W):
        flip_yz = torch.eye(4, device=C2W.device, dtype=C2W.dtype)
        flip_yz[1, 1] = -1
        flip_yz[2, 2] = -1
        bs = C2W.shape[0]
        flip_yz = flip_yz.unsqueeze(0).repeat(bs, 1, 1)
        C2W = torch.matmul(C2W, flip_yz)
        return C2W

    def convert_normal_to_cam_space(self, normal: torch.Tensor, c2w):
        """
        normal:[BN, 3]
        c2w: [B, 4, 4]
        reuturn : [BN, 3]
        """
        assert normal.ndim == 4
        batch_size, h, w, _ = normal.shape
        # normal = normal.reshape(batch_size, -1, 3, 1)

        w2c = torch.inverse(c2w)

        # normal = normal.reshape(batch_size, -1, 3, 1)  # [b, n, 3, 1]
        # normal = torch.matmul(w2c[:, :3, :3].unsqueeze(1), normal).reshape(-1, 3)  # [b, 1, 3, 3], [b, n, 3, 1]

        normal = normal.reshape(batch_size, -1, 3)  # [b, n, 3]
        rotate: Float[Tensor, "B 4 4"] = w2c[..., :3, :3]
        camera_normal = normal @ rotate.permute(0, 2, 1)
        # pixel space flip axis so we need built negative y-axis normal
        flip_x = torch.eye(3).to(w2c)
        flip_x[0, 0] = -1

        camera_normal = camera_normal @ flip_x[None, ...]
        camera_normal = camera_normal.reshape(-1, 3)

        # camera_normal = camera_normal.reshape(batch_size, h, w, 3)
        return camera_normal

    def depth_to_disparity_nd_laion(
        self, depth, valid_mask, target_min=50, target_max=255, pad_value=10
    ):
        target_min = target_min / 255.0
        target_max = target_max / 255.0
        pad_value = pad_value / 255.0

        depth_tmp = depth

        unvalid_mask = ~valid_mask

        depth_max = depth_tmp[
            valid_mask
        ].max()  # self.cfg.camera_distance_range[1] + self.cfg.radius
        depth_min = depth_tmp[valid_mask].min()  # 0.1
        depth_tmp = depth_max - depth_tmp  # reverse
        depth_tmp /= depth_max - depth_min  # [0, 1]
        depth_tmp = depth_tmp.clamp(0, 1)

        depth_tmp = depth_tmp * (target_max - target_min) + target_min
        depth_tmp[unvalid_mask] = pad_value

        return depth_tmp

    def depth_to_disparity_nd_laion(
        self,
        depth,
        valid_mask,
        target_min=50,
        target_max=255,
        pad_value=10,
        camera_distances=None,
    ):
        target_min = target_min / 255.0
        target_max = target_max / 255.0
        pad_value = pad_value / 255.0

        depth_tmp = depth

        unvalid_mask = ~valid_mask

        depth_max = depth_tmp[
            valid_mask
        ].max()  # self.cfg.camera_distance_range[1] + self.cfg.radius
        depth_min = depth_tmp[valid_mask].min()  # 0.1
        depth_tmp = depth_max - depth_tmp  # reverse
        depth_tmp /= depth_max - depth_min  # [0, 1]
        depth_tmp = depth_tmp.clamp(0, 1)

        depth_tmp = depth_tmp * (target_max - target_min) + target_min
        depth_tmp[unvalid_mask] = pad_value

        return depth_tmp

    def depth_to_disparity(
        self,
        depth,
        valid_mask,
        depth_min=None,
        depth_max=None,
        target_min=50,
        target_max=255,
        pad_value=10,
        camera_distances=None,
    ):
        depth_tmp = depth
        radius = 1.0  # 0.866 = 0.5 * sqrt(3)
        camera_distances = camera_distances.reshape(-1, 1, 1, 1)

        near = camera_distances - radius * math.sqrt(3)
        far = camera_distances + radius * math.sqrt(3)

        near_disparity = 1.0 / near
        far_disparity = 1.0 / far

        disparity = 1.0 / (depth_tmp + 1e-6)
        # disparity[disparity<=far_disparity] = far_disparity

        # print(near[0], far[0], near_disparity[0], far_disparity[0], disparity.min(), disparity.max())
        disparity = (disparity - far_disparity) / (near_disparity - far_disparity)
        # print(disparity.min(), disparity.max())

        disparity = torch.clamp(disparity, 0, 1)
        print(disparity.min(), disparity.max())

        return disparity

    def depth_normalization(
        self,
        depth,
        valid_mask,
        depth_min=None,
        depth_max=None,
        target_min=50,
        target_max=255,
        pad_value=10,
        camera_distances=None,
    ):
        depth_tmp = depth
        radius = self.cfg.depth_norm_radius  # 0.866 = 0.5 * sqrt(3)
        camera_distances = camera_distances.reshape(-1, 1, 1, 1)
        unvalid_mask = ~valid_mask

        near = camera_distances - radius * math.sqrt(3)
        far = camera_distances + radius * math.sqrt(3)

        depth_norm = (far - depth_tmp) / (far - near)

        depth_norm = torch.clamp(depth_norm, 0, 1)

        return depth_norm

    @staticmethod
    def world2camera(normal, w2c):
        rotate: Float[Tensor, "B 4 4"] = w2c[..., :3, :3]
        camera_normal = normal @ rotate.permute(0, 2, 1)
        # pixel space flip axis so we need built negative y-axis normal
        flip_x = torch.eye(3).to(w2c)
        flip_x[0, 0] = -1

        camera_normal = camera_normal @ flip_x[None, ...]

        return camera_normal

    def forward(
        self,
        mvp_mtx: Float[Tensor, "B 4 4"],
        camera_positions: Float[Tensor, "B 3"],
        light_positions: Float[Tensor, "B 3"],
        height: int,
        width: int,
        render_rgb: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        batch_size = mvp_mtx.shape[0]
        mesh = self.geometry.isosurface()

        v_pos_clip: Float[Tensor, "B Nv 4"] = self.ctx.vertex_transform(
            mesh.v_pos, mvp_mtx
        )
        rast, _ = self.ctx.rasterize(v_pos_clip, mesh.t_pos_idx, (height, width))
        mask = rast[..., 3:] > 0
        mask_aa = self.ctx.antialias(mask.float(), rast, v_pos_clip, mesh.t_pos_idx)

        out = {"opacity": mask_aa, "opacity_rerange": (mask_aa + 1.0) / 2, "mesh": mesh}

        gb_normal, _ = self.ctx.interpolate_one(mesh.v_nrm, rast, mesh.t_pos_idx)
        gb_normal = F.normalize(gb_normal, dim=-1)
        gb_normal_aa = torch.lerp(
            torch.zeros_like(gb_normal), (gb_normal + 1.0) / 2.0, mask.float()
        )
        gb_normal_aa = self.ctx.antialias(
            gb_normal_aa, rast, v_pos_clip, mesh.t_pos_idx
        )

        bg_normal = 0.5 * torch.ones_like(gb_normal_aa)
        bg_normal[:, :, :, 2] = 1.0
        bg_normal_white = torch.ones_like(gb_normal_aa)

        # using to visualized
        comp_normal_vis = torch.lerp(bg_normal, gb_normal_aa, mask.float())
        comp_normal_white_vis = torch.lerp(bg_normal_white, gb_normal_aa, mask.float())

        # convert to cam space
        w2c = c2wtow2c(kwargs["c2w"])
        batch_v_nrm = repeat(mesh.v_nrm, "n c -> b n c", b=batch_size)
        camera_normal = self.world2camera(batch_v_nrm, w2c)
        gb_normal_cam, _ = self.ctx.interpolate(camera_normal, rast, mesh.t_pos_idx)
        gb_normal_cam = F.normalize(gb_normal_cam, dim=-1)

        # -1,1-> [0,1]
        _b = (gb_normal_cam + 1.0) / 2.0

        gb_normal_cam = torch.lerp(bg_normal, _b, mask.float())
        # gb_normal_cam = torch.lerp(bg_normal, _b, mask_aa)

        gb_normal_cam_aa = self.ctx.antialias(
            gb_normal_cam, rast, v_pos_clip, mesh.t_pos_idx
        )

        # for bg white normal
        gb_normal_cam_white = torch.lerp(bg_normal_white, _b, mask.float())

        gb_normal_cam_aa_white = self.ctx.antialias(
            gb_normal_cam_white, rast, v_pos_clip, mesh.t_pos_idx
        )

        # for depth
        batch_v_pos = repeat(mesh.v_pos, "n c -> b n c", b=batch_size)
        camera_v_pos = homo_proj(batch_v_pos, w2c)
        depth = -camera_v_pos[..., 2:].contiguous()
        depth, _ = self.ctx.interpolate(depth, rast, mesh.t_pos_idx)

        depth_raw = depth

        print(
            depth_raw[depth_raw > 0].detach().cpu().numpy().min(),
            depth_raw[depth_raw > 0].detach().cpu().numpy().max(),
            depth.detach().cpu().numpy().min(),
            depth.detach().cpu().numpy().max(),
        )

        disparity = self.depth_normalization(
            depth,
            valid_mask=mask,
            target_min=50,
            pad_value=10,
            camera_distances=kwargs["camera_distances"],
        )

        disparity = torch.lerp(torch.zeros_like(disparity), disparity, mask.float())

        # [0, 1]
        disparity = self.ctx.antialias(disparity, rast, v_pos_clip, mesh.t_pos_idx)

        out["disparity"] = disparity

        out.update(
            {
                "comp_normal": gb_normal_aa,
                "comp_normal_vis": comp_normal_vis,
                "comp_normal_white_vis": comp_normal_white_vis,
                "comp_normal_cam_vis": gb_normal_cam_aa,
                "comp_normal_cam_white_vis": gb_normal_cam_aa_white,
            }
        )  # in [0, 1]

        # TODO: make it clear whether to compute the normal, now we compute it in all cases
        # consider using: require_normal_computation = render_normal or (render_rgb and material.requires_normal)
        # or
        # render_normal = render_normal or (render_rgb and material.requires_normal)

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

            rgb_fg = self.material(
                viewdirs=gb_viewdirs[selector],
                positions=positions,
                light_positions=gb_light_positions[selector],
                **extra_geo_info,
                **geo_out,
            )

            gb_rgb_bg = self.background(dirs=gb_viewdirs)
            out.update({"comp_rgb_bg": gb_rgb_bg})

            mat_all = {}
            if isinstance(rgb_fg, dict):
                assert "rgb" in rgb_fg
                mat_all = rgb_fg
            else:
                mat_all["rgb"] = rgb_fg

            for k, v in mat_all.items():
                gb_fg = torch.zeros(batch_size, height, width, 3).to(v)
                gb_fg[selector] = v

                gb_mat = torch.lerp(gb_rgb_bg, gb_fg, mask.float())
                if k == "rgb" or k == "albedo":
                    gb_mat_aa = self.ctx.antialias(
                        gb_mat, rast, v_pos_clip, mesh.t_pos_idx
                    )
                else:
                    gb_mat_aa = gb_mat

                out[f"comp_{k}"] = gb_mat_aa

        return out
