import bisect
import math
import numpy as np
import pdb
import pytorch_lightning as pl
import random
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from einops import repeat
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from threestudio.data.dmtet_uncond import (RandomCameraDataModuleConfig,
                                           RandomCameraIterableDataset,
                                           RandomCameraMultiDataset,
                                           RandomMultiViewCameraStepDataset,)
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_device, get_rank
from threestudio.utils.ops import (get_mvp_matrix, get_projection_matrix,
                                   get_ray_directions, get_rays,)
from threestudio.utils.transfer import rotate_x, rotate_y, rotate_z
from threestudio.utils.typing import *


class Dream3DOrthogonalIterableDataset(RandomCameraIterableDataset):
    """dream3d 4 views condition
    DreamCamera Idea, propose by Lingteng Qiu, et.al.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.camera_sampling_stone = cfg.camera_sampling_stone
        self.max_step = cfg.train_max_step
        self.relative = cfg.relative
        self.n_views = 4  # orthogonal view
        self.global_step = 0

    def _uniform_sampling(self, a, b, size=None):
        size = self.batch_size if size is None else size
        azimuth_deg = (torch.rand(size) + 1.0 * torch.arange(size)) / size * (a - b) + b

        azimuth = azimuth_deg * math.pi / 180
        return azimuth

    def only_front_view(self):
        # warmup for front view
        return self.all_view()

    def front_with_side(self):
        return self.all_view()

    def all_view(self):
        # all-view
        rotate_y = self._uniform_sampling(180, -180)
        return rotate_y

    def _train_scene(self, strategy):
        if strategy == 0:
            return self.only_front_view()
        elif strategy == 1:
            return self.front_with_side()
        elif strategy == 2:
            return self.all_view()
        else:
            raise NotADirectoryError

    def obtain_orthogonal_view(self, batch):
        # elevation from 0~30 only one

        assert (
            self.batch_size % self.n_views == 0
        ), f"batch_size ({self.batch_size}) must be dividable by n_views ({self.n_views})!"
        real_batch_size = self.batch_size // self.n_views
        # sample elevation angles
        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]

        if random.random() < 0.5:
            # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
            elevation_deg = (
                torch.rand(real_batch_size)
                * (self.elevation_range[1] - self.elevation_range[0])
                + self.elevation_range[0]
            ).repeat_interleave(self.n_views, dim=0)
            elevation = elevation_deg * math.pi / 180
        else:
            # otherwise sample uniformly on sphere
            elevation_range_percent = [
                (self.elevation_range[0] + 90.0) / 180.0,
                (self.elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
            elevation = torch.asin(
                2
                * (
                    torch.rand(real_batch_size)
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            ).repeat_interleave(self.n_views, dim=0)
            elevation_deg = elevation / math.pi * 180.0

        azimuth_deg: Float[Tensor, "B"]
        # ensures sampled azimuth angles in a batch cover the whole range
        azimuth_deg = (
            torch.rand(real_batch_size).reshape(-1, 1)
            + torch.arange(self.n_views).reshape(1, -1)
        ).reshape(-1) / self.n_views * (
            self.azimuth_range[1] - self.azimuth_range[0]
        ) + self.azimuth_range[
            0
        ]
        azimuth = azimuth_deg * math.pi / 180

        fovy_deg: Float[Tensor, "B"] = (
            torch.rand(real_batch_size) * (self.fovy_range[1] - self.fovy_range[0])
            + self.fovy_range[0]
        ).repeat_interleave(self.n_views, dim=0)
        fovy = fovy_deg * math.pi / 180

        camera_distances: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.camera_distance_range[1] - self.camera_distance_range[0])
            + self.camera_distance_range[0]
        ).repeat_interleave(self.n_views, dim=0)

        # rendering size
        if self.relative:
            # relative depth propose by mvdream
            scale = 1 / torch.tan(0.5 * fovy)
            relative_distances = camera_distances
            camera_distances = scale * camera_distances
        else:
            print("real_distance")
            relative_distances = camera_distances

        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.batch_size, 1)

        camera_perturb: Float[Tensor, "B 3"] = (
            torch.rand(real_batch_size, 3) * 2 * self.cfg.camera_perturb
            - self.cfg.camera_perturb
        ).repeat_interleave(self.n_views, dim=0)

        camera_positions = camera_positions + camera_perturb
        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb: Float[Tensor, "B 3"] = (
            torch.randn(real_batch_size, 3) * self.cfg.center_perturb
        ).repeat_interleave(self.n_views, dim=0)
        center = center + center_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb: Float[Tensor, "B 3"] = (
            torch.randn(real_batch_size, 3) * self.cfg.up_perturb
        ).repeat_interleave(self.n_views, dim=0)
        up = up + up_perturb

        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
            + self.cfg.light_distance_range[0]
        ).repeat_interleave(self.n_views, dim=0)

        if self.cfg.light_sample_strategy == "dreamfusion":
            # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
            light_direction: Float[Tensor, "B 3"] = F.normalize(
                camera_positions
                + torch.randn(real_batch_size, 3).repeat_interleave(self.n_views, dim=0)
                * self.cfg.light_position_perturb,
                dim=-1,
            )
            # get light position by scaling light direction by light distance
            light_positions: Float[Tensor, "B 3"] = (
                light_direction * light_distances[:, None]
            )
        elif self.cfg.light_sample_strategy == "magic3d":
            # sample light direction within restricted angle range (pi/3)
            local_z = F.normalize(camera_positions, dim=-1)
            local_x = F.normalize(
                torch.stack(
                    [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                    dim=-1,
                ),
                dim=-1,
            )
            local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
            rot = torch.stack([local_x, local_y, local_z], dim=-1)
            light_azimuth = (
                torch.rand(real_batch_size) * math.pi - 2 * math.pi
            ).repeat_interleave(
                self.n_views, dim=0
            )  # [-pi, pi]
            light_elevation = (
                torch.rand(real_batch_size) * math.pi / 3 + math.pi / 6
            ).repeat_interleave(
                self.n_views, dim=0
            )  # [pi/6, pi/2]
            light_positions_local = torch.stack(
                [
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.cos(light_azimuth),
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.sin(light_azimuth),
                    light_distances * torch.sin(light_elevation),
                ],
                dim=-1,
            )
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
        else:
            raise ValueError(
                f"Unknown light sample strategy: {self.cfg.light_sample_strategy}"
            )

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * self.height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        identity_c2w: Float[Tensor, "B 3 4"] = torch.eye(4).to(c2w)[None, :3]
        _, rays_d_tmp = get_rays(directions, identity_c2w, keepdim=True)
        depth_scale = torch.abs(rays_d_tmp[..., 2:])

        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "rays_depth_scale": depth_scale,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "relative_distances": relative_distances,
            "height": self.height,
            "width": self.width,
        }

    def obtain_singleview(self, batch):
        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]
        if random.random() < 0.5:
            # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
            elevation_deg = (
                torch.rand(self.batch_size)
                * (self.elevation_range[1] - self.elevation_range[0])
                + self.elevation_range[0]
            )
            elevation = elevation_deg * math.pi / 180
        else:
            # otherwise sample uniformly on sphere
            elevation_range_percent = [
                (self.elevation_range[0] + 90.0) / 180.0,
                (self.elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
            elevation = torch.asin(
                2
                * (
                    torch.rand(self.batch_size)
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            )
            elevation_deg = elevation / math.pi * 180.0

        # sample azimuth angles from a uniform distribution bounded by azimuth_range
        azimuth_deg: Float[Tensor, "B"]

        if self.cfg.batch_uniform_azimuth:
            # ensures sampled azimuth angles in a batch cover the whole range
            idx = bisect.bisect_left(self.camera_sampling_stone, self.global_step)
            azimuth = self._train_scene(idx)
            azimuth_deg = azimuth / math.pi * 180.0
        else:
            # simple random sampling
            azimuth_deg = (
                torch.rand(self.batch_size)
                * (self.azimuth_range[1] - self.azimuth_range[0])
                + self.azimuth_range[0]
            )
            azimuth = azimuth_deg * math.pi / 180

        # sample distances from a uniform distribution bounded by distance_range
        camera_distances: Float[Tensor, "B"] = (
            torch.rand(self.batch_size)
            * (self.camera_distance_range[1] - self.camera_distance_range[0])
            + self.camera_distance_range[0]
        )

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.batch_size, 1)

        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb: Float[Tensor, "B 3"] = (
            torch.rand(self.batch_size, 3) * 2 * self.cfg.camera_perturb
            - self.cfg.camera_perturb
        )
        camera_positions = camera_positions + camera_perturb
        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb: Float[Tensor, "B 3"] = (
            torch.randn(self.batch_size, 3) * self.cfg.center_perturb
        )
        center = center + center_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb: Float[Tensor, "B 3"] = (
            torch.randn(self.batch_size, 3) * self.cfg.up_perturb
        )
        up = up + up_perturb

        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: Float[Tensor, "B"] = (
            torch.rand(self.batch_size) * (self.fovy_range[1] - self.fovy_range[0])
            + self.fovy_range[0]
        )
        fovy = fovy_deg * math.pi / 180

        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = (
            torch.rand(self.batch_size)
            * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
            + self.cfg.light_distance_range[0]
        )

        if self.cfg.light_sample_strategy == "dreamfusion":
            # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
            light_direction: Float[Tensor, "B 3"] = F.normalize(
                camera_positions
                + torch.randn(self.batch_size, 3) * self.cfg.light_position_perturb,
                dim=-1,
            )
            # get light position by scaling light direction by light distance
            light_positions: Float[Tensor, "B 3"] = (
                light_direction * light_distances[:, None]
            )
        elif self.cfg.light_sample_strategy == "magic3d":
            # sample light direction within restricted angle range (pi/3)
            local_z = F.normalize(camera_positions, dim=-1)
            local_x = F.normalize(
                torch.stack(
                    [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                    dim=-1,
                ),
                dim=-1,
            )
            local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
            rot = torch.stack([local_x, local_y, local_z], dim=-1)
            light_azimuth = (
                torch.rand(self.batch_size) * math.pi - 2 * math.pi
            )  # [-pi, pi]
            light_elevation = (
                torch.rand(self.batch_size) * math.pi / 3 + math.pi / 6
            )  # [pi/6, pi/2]
            light_positions_local = torch.stack(
                [
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.cos(light_azimuth),
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.sin(light_azimuth),
                    light_distances * torch.sin(light_elevation),
                ],
                dim=-1,
            )
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
        else:
            raise ValueError(
                f"Unknown light sample strategy: {self.cfg.light_sample_strategy}"
            )

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * self.height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)

        identity_c2w: Float[Tensor, "B 3 4"] = torch.eye(4).to(c2w)[None, :3]
        _, rays_d_tmp = get_rays(directions, identity_c2w, keepdim=True)

        depth_scale = torch.abs(rays_d_tmp[..., 2:])

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far

        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        # normal rotation for fantasia3D, we find it is no useful sometimes
        normal_container = []
        rotate_y1 = np.random.uniform(0, np.pi * 2)
        rotate_x1 = np.random.uniform(-np.pi, np.pi)
        normal_rotate = rotate_y(rotate_y1) @ rotate_x(rotate_x1)
        normal_container.append(normal_rotate[None])

        self.global_step += 1
        condition_cameras = self.get_condition_cameras()

        # side-view
        condition_cameras_side_a = self.get_condition_cameras(
            azimuth=self.cfg.cond_view[2]
        )
        condition_cameras_side_b = self.get_condition_cameras(
            azimuth=self.cfg.cond_view[1]
        )

        # all-view
        condition_cameras_all_a = self.get_condition_cameras(
            azimuth=self.cfg.cond_view[4]
        )
        condition_cameras_all_b = self.get_condition_cameras(
            azimuth=self.cfg.cond_view[3]
        )

        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "normal_rotate": torch.cat(normal_container, dim=0),
            "mvp_mtx": mvp_mtx,
            "rays_depth_scale": depth_scale,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": self.height,
            "width": self.width,
            "condition_cameras": condition_cameras,
            "condition_cameras_side_l": condition_cameras_side_b,
            "condition_cameras_side_r": condition_cameras_side_a,
            "condition_cameras_all_l": condition_cameras_all_b,
            "condition_cameras_all_r": condition_cameras_all_a,
        }

    def collate(self, batch) -> Dict[str, Any]:
        # sample elevation angles

        # single_cond =self.obtain_singleview(batch)

        orthogonal_cond = self.obtain_orthogonal_view(batch)

        return orthogonal_cond

    def merge_condition_camera(self, cond_a, cond_b, keys):
        merge_camera = {}
        for key in keys:
            if isinstance(cond_a[key], Tensor):
                merge_camera[key] = torch.cat(
                    [cond_a[key][:, None], cond_b[key][:, None]], dim=1
                )
            else:
                merge_camera[key] = cond_a[key]
        return merge_camera

    def get_condition_cameras(self, elev=15.0, distance=3.0, azimuth=0.0):
        """for zero123, condition camera position"""
        azimuth_deg: Float[Tensor, "B"] = (torch.Tensor([azimuth])).float()
        camera_distances: Float[Tensor, "B"] = (torch.Tensor([distance])).float()
        elevation_deg: Float[Tensor, "B"] = (torch.Tensor([elev])).float()

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.cfg.eval_batch_size, 1)

        fovy_deg: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_fovy_deg
        )

        fovy = fovy_deg * math.pi / 180
        light_positions: Float[Tensor, "B 3"] = camera_positions

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = (
            0.5 * self.cfg.eval_height / torch.tan(0.5 * fovy)
        )
        directions_unit_focal = get_ray_directions(
            H=self.cfg.eval_height, W=self.cfg.eval_width, focal=1.0
        )
        directions: Float[Tensor, "B H W 3"] = directions_unit_focal[
            None, :, :, :
        ].repeat(
            1, 1, 1, 1
        )  # n_view = 1
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.cfg.eval_width / self.cfg.eval_height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        identity_c2w: Float[Tensor, "B 3 4"] = torch.eye(4).to(c2w)[None, :3]
        identity_c2w = repeat(identity_c2w, "1 w h -> b w h", b=c2w.shape[0])
        _, rays_d_tmp = get_rays(directions, identity_c2w, keepdim=True)
        depth_scale = torch.abs(rays_d_tmp[..., 2:])

        return {
            "index": -1,  # -1 means condition camera condition space
            "rays_o": rays_o,
            "rays_d": rays_d,
            "c2w": c2w,
            "rays_depth_scale": depth_scale,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "light_positions": light_positions,
            "elevation": elevation,
            "azimuth": azimuth,
            "camera_distances": camera_distances,
        }


@register("Dream3D-Orthogonal")
class Dream3DOrthogonalDataModule(pl.LightningDataModule):
    cfg: RandomCameraDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(RandomCameraDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = Dream3DOrthogonalIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            # self.val_dataset = RandomCameraDataset(self.cfg, "val")
            if not self.cfg.multiview_eval:
                self.val_dataset = RandomCameraStepDataset(self.cfg, "val")
            else:
                self.val_dataset = RandomMultiViewCameraStepDataset(self.cfg, "val")

        if stage in [None, "test", "predict"]:
            if not self.cfg.multiview_eval:
                self.test_dataset = RandomCameraDataset(self.cfg, "test")
            else:
                self.test_dataset = RandomCameraMultiDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )
