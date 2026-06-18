import open3d as o3d
import numpy as np
import os
from plyfile import PlyData

# --- config ---
SCENE_PATH = "data/scene/scene_splat.ply"
OBJECTS_PATHS = {
    "object1": "data/object1/object1_cut.ply", 
    "object2": "data/object2/object2_cut.ply"
}

# SETUP
active_obj_idx = 0
objects_list = []
objects_names = []
transforms = {} 

MOVE_STEP = 0.05
ROT_STEP = 5.0 * (np.pi / 180.0) 

def matrix_to_quaternion(R):
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        S = np.sqrt(tr+1.0) * 2
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        w = (R[2,1] - R[1,2]) / S
        x = 0.25 * S
        y = (R[0,1] + R[1,0]) / S
        z = (R[0,2] + R[2,0]) / S
    elif (R[1,1] > R[2,2]):
        S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        w = (R[0,2] - R[2,0]) / S
        x = (R[0,1] + R[1,0]) / S
        y = 0.25 * S
        z = (R[1,2] + R[2,1]) / S
    else:
        S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        w = (R[1,0] - R[0,1]) / S
        x = (R[0,2] + R[2,0]) / S
        y = (R[1,2] + R[2,1]) / S
        z = 0.25 * S
    return np.array([w, x, y, z])

def load_and_clean_pcd(path, center_it=False):
    if not os.path.exists(path): return None
    ply = PlyData.read(path)
    v = ply['vertex']
    pts = np.stack([v['x'], v['y'], v['z']], axis=1)
    try:
        c = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=1)
        c = 0.5 + 0.28209 * c
        c = np.clip(c, 0, 1)
    except:
        c = np.ones_like(pts) * 0.5
    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]
    c = c[mask]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(c)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
    
    if center_it:
        center = cl.get_center()
        cl.translate(-center)
    return cl

def get_active_obj():
    return objects_list[active_obj_idx], objects_names[active_obj_idx]

# --- TRANSlATION ---
def move_generic(vis, vec):
    obj, name = get_active_obj()
    obj.translate(vec)
    transforms[name]["pos"] += np.array(vec)
    vis.update_geometry(obj)
    return False

# --- ROTATION ---
def rotate_generic(vis, axis_idx, sign):
    obj, name = get_active_obj()
    center = obj.get_center()
    
    angles = [0, 0, 0]
    angles[axis_idx] = sign * ROT_STEP
    R_inc = o3d.geometry.get_rotation_matrix_from_xyz(angles)
    
    obj.rotate(R_inc, center=center)
    
    transforms[name]["matrix"] = R_inc @ transforms[name]["matrix"]
    
    vis.update_geometry(obj)
    return False

def move_x_pos(vis): return move_generic(vis, [MOVE_STEP, 0, 0])
def move_x_neg(vis): return move_generic(vis, [-MOVE_STEP, 0, 0])
def move_y_pos(vis): return move_generic(vis, [0, MOVE_STEP, 0])
def move_y_neg(vis): return move_generic(vis, [0, -MOVE_STEP, 0])
def move_z_pos(vis): return move_generic(vis, [0, 0, MOVE_STEP])
def move_z_neg(vis): return move_generic(vis, [0, 0, -MOVE_STEP])

def rot_y_pos(vis): return rotate_generic(vis, 1, 1)
def rot_y_neg(vis): return rotate_generic(vis, 1, -1)
def rot_x_pos(vis): return rotate_generic(vis, 0, 1)
def rot_x_neg(vis): return rotate_generic(vis, 0, -1)
def rot_z_pos(vis): return rotate_generic(vis, 2, 1)
def rot_z_neg(vis): return rotate_generic(vis, 2, -1)

# --- SELECTING AND PRINT ---
def select_obj_1(vis):
    global active_obj_idx
    active_obj_idx = 0
    print(f"Active: {objects_names[0]}")
    return False

def select_obj_2(vis):
    global active_obj_idx
    if len(objects_list) > 1:
        active_obj_idx = 1
        print(f"Active: {objects_names[1]}")
    return False

def print_result(vis):
    print("\n" + "="*60)
    print("copy and paste in generate_trajectory.py:")
    print("="*60)
    print("FINAL_TRANSFORMS = {")
    for name, data in transforms.items():
        p = data["pos"]
        q = matrix_to_quaternion(data["matrix"]) # [w, x, y, z]
        
        print(f"    '{name}': {{")
        print(f"        'pos': [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}],")
        print(f"        'rot_quat': [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")
        print(f"    }},")
    print("}")
    print("="*60 + "\n")
    return False

def main():
    print("Control scene:")
    print("---------------------------------------")
    print(" [ 1 / 2 ] -> SELECT OBJECT")
    print(" [ W / A / S / D ] + [ R / F ] -> Move")
    print(" [ I / J / K / L ] + [ U / O ] -> Rotate")
    print(" [ SPACE ] -> PRINT COORDINATES")
    print("---------------------------------------")

    global objects_list, objects_names, transforms
    
    scene = load_and_clean_pcd(SCENE_PATH, center_it=False)

    for name, path in OBJECTS_PATHS.items():
        p = load_and_clean_pcd(path, center_it=True)
        if p:
            objects_list.append(p)
            objects_names.append(name)
            transforms[name] = {"pos": np.array([0.0, 0.0, 0.0]), "matrix": np.eye(3)}

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=1280, height=720)
    
    vis.add_geometry(scene)
    for obj in objects_list:
        vis.add_geometry(obj)

    ctr = vis.get_view_control()
    if os.path.exists("view.json"):
        ctr.convert_from_pinhole_camera_parameters(o3d.io.read_pinhole_camera_parameters("view.json"))
    else:
        ctr.set_lookat([-0.5, 1.7, -0.2])
        ctr.set_zoom(0.1)

    vis.register_key_callback(87, move_z_pos) 
    vis.register_key_callback(83, move_z_neg)
    vis.register_key_callback(65, move_x_neg) 
    vis.register_key_callback(68, move_x_pos) 
    vis.register_key_callback(82, move_y_pos) 
    vis.register_key_callback(70, move_y_neg) 

    # ROTATION
    vis.register_key_callback(73, rot_x_pos) # I
    vis.register_key_callback(75, rot_x_neg) # K
    vis.register_key_callback(74, rot_y_neg) # J
    vis.register_key_callback(76, rot_y_pos) # L
    vis.register_key_callback(85, rot_z_neg) # U
    vis.register_key_callback(79, rot_z_pos) # O

    vis.register_key_callback(49, select_obj_1)
    vis.register_key_callback(50, select_obj_2)
    vis.register_key_callback(32, print_result)

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()