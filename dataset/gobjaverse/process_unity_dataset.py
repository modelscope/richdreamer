# -*- coding: utf-8 -*-

import glob
import cv2
import json
import numpy as np
import pdb
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

os.makedirs("./normal_visualized/",exist_ok=True)
os.makedirs("./unity_system/",exist_ok=True)

normal_handler = './campos_512_v4/{:05d}/{:05d}_nd.exr'
json_handler = './campos_512_v4/{:05d}/{:05d}.json'
normal_list = [normal_handler.format(i,i) for i in range(40)]
json_list = [json_handler.format(i,i) for i in range(40)]

def read_camera_matrix_single(json_file):

    with open(json_file, 'r', encoding='utf8') as reader:
        json_content = json.load(reader)

    '''
    camera_matrix = np.eye(4)
    camera_matrix[:3, 0] = np.array(json_content['x'])
    camera_matrix[:3, 1] = -np.array(json_content['y'])
    camera_matrix[:3, 2] = -np.array(json_content['z'])
    camera_matrix[:3, 3] = np.array(json_content['origin'])
    '''

    # suppose is true
    camera_matrix = np.eye(4)
    camera_matrix[:3, 0] = np.array(json_content['x'])
    camera_matrix[:3, 1] = np.array(json_content['y'])
    camera_matrix[:3, 2] = np.array(json_content['z'])
    camera_matrix[:3, 3] = np.array(json_content['origin'])


    return camera_matrix


def unity2blender(normal):
    normal_clone = normal.copy()
    normal_clone[...,0] = -normal[...,-1]
    normal_clone[...,1] = -normal[...,0]
    normal_clone[...,2] = normal[...,1]

    return normal_clone

def blender2midas(img):
    '''Blender: rub
    midas: lub
    '''
    img[...,0] = -img[...,0]
    img[...,1] = -img[...,1]
    img[...,-1] = -img[...,-1]
    return img

for normal in normal_list:
    assert os.path.exists(normal), normal
for json_path in json_list:
    assert os.path.exists(json_path), json_path


for idx, (normal_path, camera_json) in enumerate(zip(normal_list, json_list)):

    normald = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    normal = normald[...,:3]
    normal_norm = (np.linalg.norm(normal, 2, axis=-1, keepdims= True))
    # depth has some problems
    normal = normal / normal_norm
    normal = np.nan_to_num(normal,nan=-1.)


    # unity2blender
    world_normal = unity2blender(normal)

    cond_c2w = read_camera_matrix_single(camera_json)
    view_cn = blender2midas(world_normal@ (cond_c2w[:3,:3]))
    view_cn = (view_cn+1.)/2. * 255
    view_cn = np.asarray(np.clip(view_cn, 0, 255), np.uint8)

    z_dir = view_cn[...,-1]
    mask = z_dir < 127

    view_cn = view_cn[..., ::-1]
    visual_mask = view_cn * mask[...,None]

    cv2.imwrite(os.path.join("./unity_system/", "{:04d}.png".format(idx)), view_cn)
    cv2.imwrite(os.path.join("./unity_system/", "visual_mask_{:04d}.png".format(idx)), visual_mask)

