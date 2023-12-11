import numpy as np
import os
import sys
import trimesh


def load_mesh(mesh_file, y_up=False, geo_only=False, scale=1.0):
    geo_only = bool(geo_only)
    y_up = bool(y_up)

    mesh = trimesh.load(mesh_file, force="mesh")

    vertices = np.asarray(mesh.vertices)  # [N, 3]

    if y_up:
        # x-axis -90 rotation
        from scipy.spatial.transform import Rotation as R

        T_ = R.from_euler("x", -90, degrees=True).as_matrix()
        vertices = np.matmul(T_, vertices.T).T

    mesh.vertices = vertices

    if geo_only:
        mesh = trimesh.Trimesh(mesh.vertices, mesh.faces)

    return mesh


mesh_file = sys.argv[1]
if len(sys.argv) >= 3:
    out_dir = sys.argv[2]
else:
    out_dir = os.path.join(os.path.dirname(mesh_file), "rotate")

os.makedirs(out_dir, exist_ok=True)

mesh = load_mesh(mesh_file, y_up=True)

mesh.export(os.path.join(out_dir, "model.obj"))


# mesh_dirs = sys.argv[1]

# for mesh_d in os.listdir(mesh_dirs):
#     sub_mesh_d = os.path.join(mesh_dirs, mesh_d)
#     mesh_file = os.path.join(sub_mesh_d, "save/it10000-export/model.obj")
#     if os.path.exists(mesh_file):
#         print(mesh_file)
#         out_dir = os.path.join(os.path.dirname(mesh_file), "rotate")
#         os.makedirs(out_dir, exist_ok=True)
#         mesh = load_mesh(mesh_file, y_up=True)
#         mesh.export(os.path.join(out_dir, "model.obj"))
#     else:
#         print(f"do not exist:{mesh_file}")
