# -*- coding: utf-8 -*-

import glob
import cv2
import json
import numpy as np
import pdb
import os


normal_list = sorted(glob.glob('./blender_data/*_normal.png'))
camera_list = sorted(glob.glob('./blender_data/*.json'))



def blender2midas(img):
    '''Blender: rub
    midas: lub
    '''
    img[...,0] = -img[...,0]
    img[...,1] = -img[...,1]
    img[...,-1] = -img[...,-1]
    return img



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



os.makedirs('./blender_system',exist_ok= True)


for idx, (normal_path, camera_json) in enumerate(zip(normal_list, camera_list)):
    normal = cv2.imread(normal_path)
    # to xyz channel
    normal = normal[..., ::-1]
    world_normal = (normal.astype(np.float32)/255. * 2.) - 1

    cond_c2w = read_camera_matrix_single(camera_json)
    # identity map
    view_cn = blender2midas(world_normal@ (cond_c2w[:3,:3]))

    view_cn = (view_cn+1.)/2. * 255
    view_cn = np.asarray(np.clip(view_cn, 0, 255), np.uint8)
    z_dir = view_cn[...,-1]
    mask = z_dir < 127
    view_cn = view_cn[..., ::-1]

    visual_mask = view_cn * mask[...,None]


    cv2.imwrite(os.path.join("./blender_system/", "{:04d}.png".format(idx)), view_cn)
    cv2.imwrite(os.path.join("./blender_system/", "visual_mask_{:04d}.png".format(idx)), visual_mask)


