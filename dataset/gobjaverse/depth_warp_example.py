# -*- coding: utf-8 -*-
import sys
sys.path.append('./')
from typing import Dict
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
import torchvision
from einops import rearrange
import pytorch_lightning as pl
import copy
import csv
import cv2
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
import os, sys
import math
from torch.utils.data.distributed import DistributedSampler
import albumentations
import time
from tqdm import tqdm
import torch.nn.functional as F
import pdb
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from ctypes import CDLL, c_void_p, c_int,c_int32, c_float,c_bool
quick_zbuff = CDLL("./lib/build/zbuff.so")
DEBUG=False




def get_coordinate_xy(coord_shape, device):
    """get meshgride coordinate of x, y and the shape is (B, H, W)"""
    bs, height, width = coord_shape
    y_coord, x_coord = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=device),\
                                       torch.arange(0, width, dtype=torch.float32, device=device)])
    y_coord, x_coord = y_coord.contiguous(), x_coord.contiguous()
    y_coord, x_coord = y_coord.unsqueeze(0).repeat(bs, 1, 1), \
                       x_coord.unsqueeze(0).repeat(bs, 1, 1)

    return x_coord, y_coord



def read_dnormal(normald_path, cond_pos):
    cond_cam_dis = np.linalg.norm(cond_pos, 2)

    near = 0.867 #sqrt(3) * 0.5
    near_distance = cond_cam_dis - near

    normald = cv2.imread(normald_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth = normald[...,3:]

    depth[depth<near_distance] = 0

    return depth


def get_intr(target_im):
    h, w = target_im.shape[:2]

    fx = fy = 1422.222
    res_raw = 1024
    f_x = f_y = fx * h / res_raw
    K = torch.tensor([f_x, 0, w / 2, 0, f_y, h / 2, 0, 0, 1]).reshape(3, 3)
    # print("intr: ", K)
    return K


def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return torch.from_numpy(C2W)



def read_camera_matrix_single(json_file):
    with open(json_file, 'r', encoding='utf8') as reader:
        json_content = json.load(reader)

    # NOTE that different from unity2blender experiments.
    camera_matrix = np.eye(4)
    camera_matrix[:3, 0] = np.array(json_content['x'])
    camera_matrix[:3, 1] = -np.array(json_content['y'])
    camera_matrix[:3, 2] = -np.array(json_content['z'])
    camera_matrix[:3, 3] = np.array(json_content['origin'])


    '''
    camera_matrix = np.eye(4)
    camera_matrix[:3, 0] = np.array(json_content['x'])
    camera_matrix[:3, 1] = np.array(json_content['y'])
    camera_matrix[:3, 2] = np.array(json_content['z'])
    camera_matrix[:3, 3] = np.array(json_content['origin'])
    # print(camera_matrix)
    '''

    return camera_matrix


def read_w2c(camera):
    tm = camera
    tm = np.asarray(tm)

    cam_pos = tm[:3, 3:]
    world2cam = np.zeros_like(tm)
    world2cam[:3, :3] = tm[:3,:3].transpose()
    world2cam[:3,3:] = -tm[:3,:3].transpose() @ tm[:3,3:]
    world2cam[-1, -1] = 1

    return world2cam, np.linalg.norm(cam_pos, 2 , axis=0)



def get_camera_pos(camera):
    tm = camera['transform_matrix']
    tm = np.asarray(tm)

    cam_pos = tm[:3, 3:]
    return cam_pos


def to_torch_tensor(input):
    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input)
    return input


def image_warping_v1(target_img, ref_img, K, c2w_t, c2w_r, target_depth, ref_depth, scale_factor=1.0, device=torch.device("cpu"), save_root=None):

    # normalized input imgs [-1, 1]
    target_img = target_img.astype(np.float32)
    ref_img = ref_img.astype(np.float32)
    target_img = target_img /255. * 2. -1
    ref_img = ref_img / 255. * 2. - 1

    with torch.no_grad():

        ref_K = K

        # target_img: [H, W, 3], target_depth:[H, W, 1], K:[3, 3], T_t2r:[4, 4]
        t_img = to_torch_tensor(target_img).permute(2, 0, 1).unsqueeze(0).float().to(device) # [1, 3, H, W]
        r_img = to_torch_tensor(ref_img).permute(2, 0, 1).unsqueeze(0).float().to(device) # [1, 3, H, W]


        # T_t2r = to_torch_tensor(T_t2r).unsqueeze(0).float().to(device) # [1, 4, 4]

        c2w_t = to_torch_tensor(c2w_t).unsqueeze(0).float().to(device)
        c2w_r = to_torch_tensor(c2w_r).unsqueeze(0).float().to(device)
        target_depth = to_torch_tensor(target_depth).permute(2, 0, 1).float().to(device) #[1, H, W]
        ref_depth = to_torch_tensor(ref_depth).permute(2, 0, 1).float().to(device) #[1, H, W]

        K = to_torch_tensor(K).unsqueeze(0).float().to(device)  # [1, 3, 3]
        ref_K = to_torch_tensor(ref_K).unsqueeze(0).float().to(device)  # [1, 3, 3]

        t_pose = {"intr": K, "extr": torch.inverse(c2w_t)}
        r_pose = {"intr": K, "extr": torch.inverse(c2w_r)}


        ref_img_warped, ref_depth_warpped = image_warpping_reproj(depth_ref=ref_depth,
                              depth_src=None,
                              ref_pose=r_pose,
                              src_pose=t_pose,
                              img_ref=r_img)



        # only using in debug
        if save_root is not None:
            os.makedirs(save_root, exist_ok=True)


            img_w = ref_img_warped[0].permute(1,2,0).detach().cpu().numpy()
            t_img = t_img[0].permute(1,2,0).detach().cpu().numpy()
            r_img = r_img[0].permute(1,2,0).detach().cpu().numpy()


            img_blend = 0.5 * t_img + 0.5 * img_w

            save_name = os.path.join(save_root, f"blend.jpg")
            img_vis = np.hstack([t_img, img_w, r_img, img_blend, 0.5 * t_img + 0.5 * r_img])

            cv2.imwrite(save_name, np.clip((img_vis + 1) / 2 * 255, 0, 255).astype(np.uint8)[:, :, (2, 1, 0)])

        return ref_img_warped



def zbuff_check(xyz,bs,height,width):

    p_xyz, depth = xyz[:, :2] / (xyz[:, 2:3].clamp(min=1e-10)),xyz[:,2:3]
    x_src = p_xyz[:, 0].view([bs, 1, -1 ]).round()  #[B, H, W]
    y_src = p_xyz[:, 1].view([bs, 1, -1]).round()

    valid_mask_0= torch.logical_and(x_src<width, y_src<height).view(-1)
    valid_mask_1= torch.logical_and(x_src>=0, y_src>=0).view(-1)
    valid_mask = torch.logical_and(valid_mask_0, valid_mask_1)


    x_src = x_src.clamp(0, width - 1).long()
    y_src = y_src.clamp(0, height - 1).long()

    buffs= -torch.ones((height,width)).to(xyz)
    z_buffs= -torch.ones((height,width)).to(xyz)

    # zbuff_check

    src_x = x_src.view(-1).numpy().astype(np.int32)
    src_y = y_src.view(-1).numpy().astype(np.int32)
    depth = depth.view(-1).numpy().astype(np.float32)
    data_size = c_int(src_x.shape[0])
    valid_mask = valid_mask.numpy()

    buffs= buffs.numpy().astype(np.float32)
    z_buffs= z_buffs.numpy().astype(np.float32)

    h, w = z_buffs.shape


    # using C++ version
    quick_zbuff.zbuff_check(src_x.ctypes.data_as(c_void_p), src_y.ctypes.data_as(c_void_p), \
            depth.ctypes.data_as(c_void_p), data_size, valid_mask.ctypes.data_as(c_void_p), buffs.ctypes.data_as(c_void_p),\
            z_buffs.ctypes.data_as(c_void_p), h, w)



    '''
    for idx, (x, y, z) in enumerate(zip(x_src.view(-1),y_src.view(-1), depth.view(-1))):
        if not valid_mask[idx]:
            continue
        if buffs[y,x] ==-1:
            buffs[y,x] =idx
            z_buffs[y,x] =z
        else:
            if z_buffs[y,x] > z:
                buffs[y,x] =idx
                z_buffs[y,x] =z
    '''

    valid_buffs = torch.from_numpy(buffs[buffs!=-1])





    return valid_buffs.long()





def reproject_with_depth_batch(depth_ref, depth_src, ref_pose, src_pose, xy_coords, img_ref):
    """project the reference point cloud into the source view, then project back"""
    # img_src: [B, 3, H, W], depth:[B, H, W], extr: w2c
    img_tgt = -torch.ones_like(img_ref)

    depth_tgt = 5 * torch.ones_like(img_ref) # background setting to 5

    intrinsics_ref, extrinsics_ref = ref_pose["intr"], ref_pose["extr"]
    intrinsics_src, extrinsics_src = src_pose["intr"], src_pose["extr"]



    bs, height, width = depth_ref.shape[:3]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = xy_coords  # (B, H, W)
    x_ref, y_ref = x_ref.view([bs, 1, -1]), y_ref.view([bs, 1, -1])  # (B, 1, H*W)
    ref_indx = (y_ref * height+ x_ref).long().squeeze()

    depth_mask = torch.logical_not(((depth_ref.view([bs, 1, -1]))[..., ref_indx] ==5.))[0,0]
    x_ref = x_ref[..., depth_mask]
    y_ref = y_ref[..., depth_mask]

    depth_ref = depth_ref.view(bs, 1, -1)
    depth_ref = depth_ref[..., depth_mask]

    # reference 3D space, depth_view condition
    xyz_ref = torch.matmul(torch.inverse(intrinsics_ref), torch.cat([x_ref, y_ref, torch.ones_like(x_ref)], dim=1) * depth_ref.view([bs, 1, -1]))  # (B, 3, H*W)
    # source 3D space
    xyz_src = torch.matmul(torch.matmul(extrinsics_src, torch.inverse(extrinsics_ref)), \
                           torch.cat([xyz_ref, torch.ones_like(x_ref)], dim=1))[:, :3]


    # source view x, y
    k_xyz_src = torch.matmul(intrinsics_src, xyz_src)
    zbuff_idx = zbuff_check(k_xyz_src, bs, height, width)
    x_ref = x_ref[..., zbuff_idx]
    y_ref = y_ref[..., zbuff_idx]
    depth_ref= depth_ref[..., zbuff_idx]
    k_xyz_src = k_xyz_src[...,zbuff_idx]
    xy_src = k_xyz_src[:, :2] / (k_xyz_src[:, 2:3].clamp(min=1e-10))  # (B, 2, H*W)
    src_depth = k_xyz_src[:, 2:3]


    x_src = xy_src[:, 0].view([bs, 1, -1 ]).round()  #[B, H, W]
    y_src = xy_src[:, 1].view([bs, 1, -1]).round()

    # x_src_norm = x_src / ((width - 1) / 2) - 1
    # y_src_norm = y_src / ((height - 1) / 2) - 1
    # xy_src_norm = torch.stack([x_src_norm, y_src_norm], dim=3)
    x_src = x_src.clamp(0, width - 1).long()
    y_src = y_src.clamp(0, height - 1).long()

    img_tgt_tmp = img_tgt.permute(0, 2, 3, 1) #[B, H, W, 3]
    depth_tgt_tmp = depth_tgt.permute(0, 2, 3, 1)[...,0] #[B, H, W, 1]
    img_ref_tmp = img_ref.permute(0, 2, 3, 1) #[B, H, W, 3]


    B, _, H, W = img_ref.shape
    bs_tensor = torch.arange(B, dtype=x_src.dtype, device=x_src.device).unsqueeze(1).unsqueeze(1).repeat(1, H, W)

    bs_tensor = torch.zeros_like(x_ref).long()
    x_ref = x_ref.long()
    y_ref = y_ref.long()

    img_tgt_tmp[bs_tensor, y_src, x_src] = img_ref_tmp[bs_tensor, y_ref, x_ref]
    img_tgt = img_tgt_tmp.permute(0, 3, 1, 2)

    depth_tgt_tmp[bs_tensor,y_src,x_src]=src_depth
    depth_tgt = depth_tgt_tmp.unsqueeze(1)



    return img_tgt, depth_tgt


def image_warpping_reproj(depth_ref, depth_src, ref_pose, src_pose,
                          img_ref, mask_ref=None,
                          thres_p_dist=15, thres_d_diff=0.1, device=torch.device("cpu"), bg_color=1.0):
    """check geometric consistency
    consider two factor:
    1.disparity < 1
    2.relative depth differ ratio < 0.001
    # warp img_src to ref


    depth_ref: depth reference
    """
    # img_src: [B, 3, H, W], depth:[B, H, W], extr: w2c, mask_ref[B, H, W]

    x_ref, y_ref = get_coordinate_xy(depth_ref.shape, device=device)  # (B, H, W)
    xy_coords = x_ref, y_ref

    img_ref_warped, depth_ref_warpped = \
        reproject_with_depth_batch(depth_ref, depth_src, ref_pose, src_pose, xy_coords, img_ref)

    img = ((img_ref[0].permute(1,2,0) +1.) /2 * 255)
    warp_img = ((img_ref_warped[0].permute(1,2,0) +1.) /2 * 255)



    return img_ref_warped, depth_ref_warpped






def warp(img_list, normald_list, json_list, cond_idx, target_idx):

    cond_img = img_list[cond_idx]
    target_img = img_list[target_idx]


    cond_camera_path= json_list[cond_idx]
    cond_view_c2w = read_camera_matrix_single(cond_camera_path)

    cond_view_pos = cond_view_c2w[:3, 3:]
    cond_world_view_depth = read_dnormal(normald_list[cond_idx], cond_view_pos)
    cond_world_view_depth = torch.from_numpy(cond_world_view_depth)
    # background is mapped to far plane e.g. 5
    cond_world_view_depth[cond_world_view_depth==0]=5.

    cond_img = cv2.imread(cond_img)

    # target parameters
    target_camera_path= json_list[target_idx]
    target_view_c2w = read_camera_matrix_single(target_camera_path)
    target_view_pos = target_view_c2w[:3, 3:]

    target_world_view_depth = read_dnormal(normald_list[target_idx], target_view_pos)
    target_world_view_depth = torch.from_numpy(target_world_view_depth)
    # background is mapped to far plane e.g. 5
    target_world_view_depth[target_world_view_depth==0]=5.

    K = get_intr(cond_world_view_depth) # fixed metric from our blender
    target_img = cv2.imread(target_img)


    cond_normal_warped = image_warping_v1(target_img, cond_img,
                    K,
                    convert_pose(target_view_c2w),
                    convert_pose(cond_view_c2w),
                    target_world_view_depth,
                    cond_world_view_depth,
                    scale_factor=1.0, device=torch.device("cpu"), save_root='./depth_warpping_exps')



if __name__ == '__main__':



    img_handler = './campos_512_v4/{:05d}/{:05d}.png'
    normald_handler = './campos_512_v4/{:05d}/{:05d}_nd.exr'
    json_handler = './campos_512_v4/{:05d}/{:05d}.json'
    img_list = [img_handler.format(i,i) for i in range(40)]
    normald_list = [normald_handler.format(i,i) for i in range(40)]
    json_list = [json_handler.format(i,i) for i in range(40)]


    warp(img_list, normald_list, json_list, int(sys.argv[1]), int(sys.argv[2]))


