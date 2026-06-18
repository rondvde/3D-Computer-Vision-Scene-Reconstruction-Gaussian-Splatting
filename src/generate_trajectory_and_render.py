#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 14:47:34 2025

@author: daviderondini

environment: Google Colab - T4 GPU
"""

# CHUNK 1 ####################################################################
import os

working_directory = '/content/drive/MyDrive/Computer Vision'
os.chdir(working_directory)
print(f"Working directory set to: {os.getcwd()}")

#CHUNK 2######################################################################
!pip install plyfile

!pip uninstall -y torch torchvision torchaudio gsplat

!pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

!pip install gsplat==1.0.0 plyfile imageio numpy

#CHUNK 3 #####################################################################
import os

ROOT = '/content/drive/MyDrive/Computer Vision'

SCENE_PLY = f"{ROOT}/data/scene/scene_splat.ply"
OBJ1_PLY  = f"{ROOT}/data/object1/object1_cut.ply"
OBJ2_PLY  = f"{ROOT}/data/object2/object2_cut.ply"

MOTION_JSON = f"{ROOT}/output/motion_script.json"
FRAME_DIR   = f"{ROOT}/frames"
VIDEO_OUT   = f"{ROOT}/output/final_movie.mp4"

os.makedirs(FRAME_DIR, exist_ok=True)

#CHUNK 4 - GENERATING TRAJECTORY AND POSITIONS OF OBJECTS ####################
import json
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

FPS = 30
DURATION = 10 
FRAMES = FPS * DURATION
OUTPUT_FILE = "output/motion_script.json"
VIEW_FILE = "view.json" 

FINAL_TRANSFORMS = {
    'object1': {
        'pos': [0.5500, 0.7000, -0.1000],
        'rot_quat': [-0.0874, -0.1028, 0.8816, -0.4523] # [w, x, y, z]
    },
    'object2': {
        'pos': [0.8000, 0.5500, -0.1500],
        'rot_quat': [-0.3056, 0.4086, 0.7498, -0.4211] # [w, x, y, z]
    },
}

def get_camera_from_file():
    if not os.path.exists(VIEW_FILE):
        raise FileNotFoundError(f"file not found")

    with open(VIEW_FILE, 'r') as f:
        data = json.load(f)
        
    extrinsic = np.array(data["extrinsic"]).reshape(4, 4).T
    c2w = np.linalg.inv(extrinsic)
    
    start_pos = c2w[:3, 3]
    rot_mtx = c2w[:3, :3]
    q_xyzw = R.from_matrix(rot_mtx).as_quat() 
    q_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]] 
    
    return start_pos, q_wxyz

def apply_rotation_wobble(base_quat_wxyz, t):
    base_xyzw = [base_quat_wxyz[1], base_quat_wxyz[2], base_quat_wxyz[3], base_quat_wxyz[0]]
    r_base = R.from_quat(base_xyzw)
    
    angle_deg = 10.0 * np.sin(t * 5.0)
    r_wobble = R.from_euler('y', angle_deg, degrees=True)
    
    r_final = r_base * r_wobble
    
    final_xyzw = r_final.as_quat()
    final_wxyz = [final_xyzw[3], final_xyzw[0], final_xyzw[1], final_xyzw[2]]
    return final_wxyz

def generate():
    data = []
    
    start_pos, start_rot = get_camera_from_file()
    
    p1 = np.array(FINAL_TRANSFORMS['object1']['pos'])
    p2 = np.array(FINAL_TRANSFORMS['object2']['pos'])
    target_pos = (p1 + p2) / 2.0
    zoom_vector = target_pos - start_pos
    
    pos1_base = FINAL_TRANSFORMS['object1']['pos']
    rot1_base = FINAL_TRANSFORMS['object1']['rot_quat']
    pos2_base = FINAL_TRANSFORMS['object2']['pos']
    rot2_base = FINAL_TRANSFORMS['object2']['rot_quat']
    
    obj2_x_at_4s = pos2_base[0] - (4.0 * 0.05)

    print(f"generation...")

    for i in range(FRAMES):
        t = i / FPS
        frame = {"frame_id": i, "camera": {}, "objects": {}}

        zoom_pct = 0.0
        if t <= 4.0:
            zoom_pct = (t / 4.0) * 0.45 
        elif t < 7.0:
            zoom_pct = 0.45 + ((t - 4.0) * 0.006)
        else:
            zoom_pct = 0.30 + 0.15 * np.sin((t - 7.0) * 20.0)

        current_cam_pos = start_pos + (zoom_vector * zoom_pct)

        frame["camera"] = {
            "position": current_cam_pos.tolist(),
            "rotation": start_rot,
            "fov": 60
        }

        s1 = 1.0 
        if t >= 7.0:
            if t < 8.0:
                s1 = 1.0 + (t - 7.0)
            else:
                s1 = 2.0
        
        frame["objects"]["object1"] = {
            "position": pos1_base, 
            "rotation": rot1_base,
            "scale": [s1, s1, s1]
        }

        pos2_curr = list(pos2_base)
        rot2_curr = list(rot2_base)
        s2 = 1.0
        
        if t < 4.0:
            pos2_curr[0] = pos2_base[0] - (t * 0.05)
            rot2_curr = apply_rotation_wobble(rot2_base, t)
            
        elif t < 8.0:
            pos2_curr[0] = obj2_x_at_4s 
            
            if t < 7.0: 
                s2 = 1.0 + 0.1 * np.sin((t - 4.0) * 15.0) 
                
            elif t >= 7.0 and t < 8.0:
                noise_intensity = 0.015
                pos2_curr[0] += np.random.uniform(-noise_intensity, noise_intensity)
                pos2_curr[1] += np.random.uniform(-noise_intensity, noise_intensity)
                pos2_curr[2] += np.random.uniform(-noise_intensity, noise_intensity)
                s2 = 1.0
                
        else:
            start_run = obj2_x_at_4s
            run_speed = 1.5 
            pos2_curr[0] = start_run + ((t - 8.0) * run_speed)

        frame["objects"]["object2"] = {
            "position": pos2_curr,
            "rotation": rot2_curr,
            "scale": [s2, s2, s2]
        }
        
        data.append(frame)
    return data

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(generate(), f, indent=4)
    print(f"file saved: {OUTPUT_FILE}")

#CHUNK 5 - RENDERING MOVIE ###################################################

import torch
import json
import imageio
import numpy as np
import os
import math
from plyfile import PlyData
from gsplat import rasterization

# --- CONFIG ---
DEVICE = "cuda"
RENDER_WIDTH = 1280
RENDER_HEIGHT = 720
SCENE_PLY = f"{ROOT}/data/scene/scene_splat.ply"
OBJ1_PLY  = f"{ROOT}/data/object1/object1_cut.ply"
OBJ2_PLY  = f"{ROOT}/data/object2/object2_cut.ply"
MOTION_JSON = f"{ROOT}/output/motion_script.json"
FRAME_DIR   = f"{ROOT}/frames"
VIDEO_OUT   = f"{ROOT}/output/final_movie.mp4"

os.makedirs(FRAME_DIR, exist_ok=True)

def quaternion_to_matrix(q):
    q = q / q.norm()
    w, x, y, z = q
    return torch.tensor([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w,     0],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w,     0],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y, 0],
        [0,                 0,                 0,                 1]
    ], device=DEVICE).float()

def transform_gaussians(means, quats, scales, translation, rotation_quat):
    R_obj = quaternion_to_matrix(torch.tensor(rotation_quat, device=DEVICE).float())[:3, :3]
    t_vec = torch.tensor(translation, device=DEVICE).float()
    transformed_means = torch.matmul(means, R_obj.T) + t_vec
    return transformed_means, quats, scales

def load_ply_gsplat(path, center_it=False):
    print(f"loading files...")
    plydata = PlyData.read(path)
    v = plydata['vertex']
    means = torch.stack((torch.tensor(v['x'], device=DEVICE), torch.tensor(v['y'], device=DEVICE), torch.tensor(v['z'], device=DEVICE)), dim=1).float()
    
    if center_it:
        centroid = torch.mean(means, dim=0)
        means = means - centroid
    
    try:
        f_dc_0 = torch.tensor(v['f_dc_0'], device=DEVICE)
        f_dc_1 = torch.tensor(v['f_dc_1'], device=DEVICE)
        f_dc_2 = torch.tensor(v['f_dc_2'], device=DEVICE)
        colors = torch.stack((f_dc_0, f_dc_1, f_dc_2), dim=1) * 0.282 + 0.5
    except:
        colors = torch.stack((torch.tensor(v['red'], device=DEVICE), torch.tensor(v['green'], device=DEVICE), torch.tensor(v['blue'], device=DEVICE)), dim=1).float() / 255.0

    scales = torch.exp(torch.stack((torch.tensor(v['scale_0'], device=DEVICE), torch.tensor(v['scale_1'], device=DEVICE), torch.tensor(v['scale_2'], device=DEVICE)), dim=1).float())
    quats = torch.stack((torch.tensor(v['rot_0'], device=DEVICE), torch.tensor(v['rot_1'], device=DEVICE), torch.tensor(v['rot_2'], device=DEVICE), torch.tensor(v['rot_3'], device=DEVICE)), dim=1).float()
    opacities = torch.sigmoid(torch.tensor(v['opacity'], device=DEVICE).float())
    return {"means": means, "colors": colors, "scales": scales, "quats": quats, "opacities": opacities}

def get_camera_matrices(cam_pos, cam_rot_quat):
    R = quaternion_to_matrix(torch.tensor(cam_rot_quat, device=DEVICE).float())[:3, :3]
    pos = torch.tensor(cam_pos, device=DEVICE).float()
    R_w2c = R.T
    t_w2c = -torch.matmul(R_w2c, pos)
    viewmat = torch.eye(4, device=DEVICE)
    viewmat[:3, :3] = R_w2c
    viewmat[:3, 3] = t_w2c
    fov_rad = math.radians(60)
    focal = RENDER_HEIGHT / (2.0 * math.tan(fov_rad / 2.0))
    K = torch.tensor([[focal, 0, RENDER_WIDTH/2], [0, focal, RENDER_HEIGHT/2], [0, 0, 1]], device=DEVICE).float()
    return viewmat.unsqueeze(0), K.unsqueeze(0)

scene_data = load_ply_gsplat(SCENE_PLY, center_it=False)
objects_data = {"object1": load_ply_gsplat(OBJ1_PLY, center_it=True), "object2": load_ply_gsplat(OBJ2_PLY, center_it=True)}

with open(MOTION_JSON) as f:
    motion_script = json.load(f)

rendered_frames = []
print(f"Rendering {len(motion_script)} frames...")

for frame in motion_script:
    idx = frame["frame_id"]
    
    # 1. JSON POS
    cam = frame["camera"]
    viewmats, Ks = get_camera_matrices(cam["position"], cam["rotation"])
    
    # 2. ASSEMBLE
    all_means = [scene_data["means"]]
    all_colors = [scene_data["colors"]]
    all_scales = [scene_data["scales"]]
    all_quats = [scene_data["quats"]]
    all_opacities = [scene_data["opacities"]]
    
    for obj_name, transf in frame.get("objects", {}).items():
        if obj_name in objects_data:
            orig = objects_data[obj_name]
            s_factor = transf["scale"][0]
            cur_scales = orig["scales"] * s_factor
            cur_means, cur_quats, _ = transform_gaussians(orig["means"] * s_factor, orig["quats"], cur_scales, transf["position"], transf["rotation"])
            all_means.append(cur_means)
            all_colors.append(orig["colors"])
            all_scales.append(cur_scales)
            all_quats.append(cur_quats)
            all_opacities.append(orig["opacities"])

    final_means = torch.cat(all_means, dim=0)
    final_colors = torch.cat(all_colors, dim=0)
    final_scales = torch.cat(all_scales, dim=0)
    final_quats = torch.cat(all_quats, dim=0)
    final_opacities = torch.cat(all_opacities, dim=0)

    # 3. RASTER
    image, _, _ = rasterization(means=final_means, quats=final_quats, scales=final_scales, opacities=final_opacities, colors=final_colors, viewmats=viewmats, Ks=Ks, width=RENDER_WIDTH, height=RENDER_HEIGHT, packed=False)
    
    image = image[0].clamp(0, 1).detach().cpu().numpy()
    image = (image * 255).astype(np.uint8)
    out_path = f"{FRAME_DIR}/frame_{idx:05d}.png"
    imageio.imwrite(out_path, image)
    rendered_frames.append(image)
    if idx % 10 == 0: print(f"   Frame {idx} fatto")

imageio.mimsave(VIDEO_OUT, rendered_frames, fps=30, format='FFMPEG')
print(f"DONE. Video: {VIDEO_OUT}")