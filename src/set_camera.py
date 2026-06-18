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

# camera sensitiveness
STEP = 0.05 

def load_and_clean_pcd(path):
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
    return cl

# --- movement functions ---
def move_left(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(-STEP, 0, 0)
    return False

def move_right(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(STEP, 0, 0)
    return False

def move_up(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(0, STEP, 0)
    return False

def move_down(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(0, -STEP, 0)
    return False

def zoom_in(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(0, 0, -STEP) 
    return False

def zoom_out(vis):
    ctr = vis.get_view_control()
    ctr.camera_local_translate(0, 0, STEP)
    return False

def save_and_quit(vis):
    print("\nsaving camera intrinsic and extrinisic...")
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters("view.json", param)
    print("saved in 'view.json'")
    vis.close()
    return False

def main():
    print("Controller activated")
    print("---------------------------------------")
    print(" [ W ] / [ S ]      -> Forward / Backward (Zoom)")
    print(" [ A ] / [ D ]      -> Left / Right")
    print(" [ Freccia SU/GIÙ ] -> Up / Down")
    print(" [ Q ]              -> Save and exit")
    print("---------------------------------------")

    full_scene = o3d.geometry.PointCloud()
    scene = load_and_clean_pcd(SCENE_PATH)
    if scene: full_scene += scene
    
    geometries = [scene]
    for path in OBJECTS_PATHS.values():
        p = load_and_clean_pcd(path)
        if p: 
            full_scene += p
            geometries.append(p)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=1280, height=720)
    
    for g in geometries:
        vis.add_geometry(g)
    
    ctr = vis.get_view_control()
    ctr.set_lookat([-0.518, 1.762, -0.188]) 
    ctr.set_front([0.0, 0.0, -0.5]) 
    ctr.set_up([0.0, -1.0, 0.0])
    ctr.set_zoom(0.1)

    vis.register_key_callback(87, zoom_in)    # W
    vis.register_key_callback(83, zoom_out)   # S
    vis.register_key_callback(65, move_left)  # A
    vis.register_key_callback(68, move_right) # D
    vis.register_key_callback(81, save_and_quit) # Q
    
    vis.register_key_callback(265, move_up)   # Arrow UP
    vis.register_key_callback(266, move_down) # Arrow DOWN

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.0

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()