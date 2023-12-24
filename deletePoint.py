import numpy as np
import trimesh
import random

mesh = trimesh.load_mesh('./data/mask.ply')

points = mesh.vertices
colors = mesh.visual.vertex_colors

num_points_to_delete = int(len(points) * 0.8)  # 删除20%的点

indices_to_delete = random.sample(range(len(points)), num_points_to_delete)

points = np.delete(points, indices_to_delete, axis=0)
colors = np.delete(colors, indices_to_delete, axis=0)

mesh.vertices = points
mesh.visual.vertex_colors = colors

mesh.export('maskLoss.ply', file_type='ply')