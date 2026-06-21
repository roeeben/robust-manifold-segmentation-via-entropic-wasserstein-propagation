import numpy as np
import pyvista as pv
from scipy import stats

from src.geometry.config import MESH_PATH, MESH_CONFIGS
from src.geometry.mesh import Mesh
from src.solvers.heat import HeatKernel
from src.solvers.sinkhorn import SinkhornSolver

def vertices_to_faces_majority_vote(vertex_labels, faces):
    """
    assign the label by the neighbors to prevent rainbow artifacts in pyvista
    """
    l0 = vertex_labels[faces[:, 0]]
    l1 = vertex_labels[faces[:, 1]]
    l2 = vertex_labels[faces[:, 2]]

    modes, _ = stats.mode(np.stack([l0, l1, l2]), axis=0, keepdims=True)  # most common value
    return modes[0]


def main(active_mesh, gamma, iterations, colors_seg):
    mesh_data = pv.read(MESH_PATH)

    pts = mesh_data.points
    if mesh_data.is_all_triangles:
        faces = mesh_data.faces.reshape(-1, 4)[:, 1:]
    else:
        tri = mesh_data.triangulate()
        pts = tri.points
        faces = tri.faces.reshape(-1, 4)[:, 1:]

    mesh = Mesh(pts, faces)

    print(f"Solving for Gamma = {gamma}")
    kernel = HeatKernel(mesh, gamma=gamma)
    solver = SinkhornSolver(mesh, kernel)

    # anchors setup
    anchors_dict = MESH_CONFIGS[active_mesh]["anchors"]
    seeds = {}
    for i, (key, v_idx) in enumerate(anchors_dict.items()):
        seeds[v_idx] = i
    num_classes = len(seeds)

    transport_plan = solver.solve(seeds, num_classes, iterations=iterations)  # [num_vertices, num_classes]
    row_sums = np.sum(transport_plan, axis=1, keepdims=True)
    transport_plan /= (row_sums + 1e-20)

    labels = np.argmax(transport_plan, axis=1)

    print("Visualizing")
    visualize_result(mesh_data, mesh, labels, seeds, colors_seg)


def visualize_result(raw_mesh_data, physics_mesh, labels, seeds, colors_seg):
    pv_mesh = pv.wrap(raw_mesh_data).copy(deep=True)
    pv_mesh.points = physics_mesh.vertices

    # annoying pyvista rainbows fix
    face_labels = vertices_to_faces_majority_vote(labels, physics_mesh.faces)
    pv_mesh.cell_data["Labels"] = face_labels

    p = pv.Plotter()
    p.add_mesh(pv_mesh, scalars="Labels", cmap=colors_seg, clim=[0, 6], show_scalar_bar=False)

    # plot the anchors as white dots
    for k, (v_idx, cls) in enumerate(seeds.items()):
        seed_pos = physics_mesh.vertices[v_idx]
        p.add_points(seed_pos, color="white", point_size=25,
                     render_points_as_spheres=True)

    # setup the camera for the SCAPE mesh
    center = pv_mesh.center
    p.camera.up = (1, 0, 0)
    p.camera.position = (center[0], center[1], center[2] + 3.0)
    p.camera.focal_point = center
    p.reset_camera()
    p.camera.zoom(1.2)

    p.show()


if __name__ == "__main__":
    active_mesh = "mesh019"
    gamma = 0.02
    iterations = 50
    colors_seg = ["red", "green", "blue", "yellow", "cyan", "purple", "white"]
    main(active_mesh=active_mesh, gamma=gamma, iterations=iterations, colors_seg=colors_seg)