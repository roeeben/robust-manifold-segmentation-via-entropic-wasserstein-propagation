import numpy as np
import pyvista as pv
from scipy import stats

from src.geometry.config import MESH_PATH, MESH_CONFIGS
from src.geometry.mesh import Mesh
from src.solvers.heat import HeatKernel
from src.solvers.sinkhorn import SinkhornSolver

def solve_for_gamma(mesh, seeds, gamma, iterations):
    print(f"Solving for gamma = {gamma}")
    kernel = HeatKernel(mesh, gamma=gamma)
    solver = SinkhornSolver(mesh, kernel)
    plan = solver.solve(seeds, len(seeds), iterations=iterations)

    row_sums = np.sum(plan, axis=1, keepdims=True)
    plan /= (row_sums + 1e-20)

    labels = np.argmax(plan, axis=1)
    confidence = np.max(plan, axis=1)
    return labels, confidence

# for the annoying pyvista rainbow bug
def vertices_to_faces_majority_vote(vertex_labels, faces):
    l0 = vertex_labels[faces[:, 0]]
    l1 = vertex_labels[faces[:, 1]]
    l2 = vertex_labels[faces[:, 2]]
    modes, _ = stats.mode(np.stack([l0, l1, l2]), axis=0, keepdims=True)
    return modes[0]


def run_gamma_analysis(active_mesh, iterations, gammas_to_test, colors_seg):
    print("Running Gamma analysis")
    mesh_data = pv.read(MESH_PATH)

    pts = mesh_data.points
    if mesh_data.is_all_triangles:
        faces = mesh_data.faces.reshape(-1, 4)[:, 1:]
    else:
        tri = mesh_data.triangulate()
        pts = tri.points
        faces = tri.faces.reshape(-1, 4)[:, 1:]

    mesh = Mesh(pts, faces)

    anchors = MESH_CONFIGS[active_mesh]["anchors"]
    seeds = {v: i for i, (k, v) in enumerate(anchors.items())}

    results = [solve_for_gamma(mesh, seeds, g, iterations) for g in gammas_to_test]

    # pyvista colorbar hack from https://github.com/pyvista/pyvista/discussions/4542
    p = pv.Plotter(
        shape=(3, len(gammas_to_test)),
        row_weights=[1, 1, 0.2],
        groups=[(2, np.s_[:])],
        border=False
    )
    p.link_views()

    base_mesh = pv.wrap(mesh_data).copy(deep=True)
    base_mesh.points = mesh.vertices

    for i, g in enumerate(gammas_to_test):
        lbl, conf = results[i]

        # Row 0: labels
        p.subplot(0, i)
        p.add_text(f"Gamma = {g}\n(Labels)", font_size=10, position="upper_right")
        m = base_mesh.copy(deep=True)
        face_lbl = vertices_to_faces_majority_vote(lbl, mesh.faces)
        m.cell_data["Labels"] = face_lbl
        p.add_mesh(m, scalars="Labels", cmap=colors_seg, clim=[0, 6], show_scalar_bar=False)

        # Row 1: confidence
        p.subplot(1, i)
        avg_conf = np.mean(conf)
        p.add_text(f"Conf (Avg: {avg_conf:.2f})", font_size=10, position="upper_right")
        m2 = base_mesh.copy(deep=True)
        m2["Confidence"] = conf
        p.add_mesh(m2, scalars="Confidence", cmap="viridis", clim=[0, 1], show_scalar_bar=False)


    p.subplot(2, 0)
    dummy_mesh = pv.PolyData([0.0, 0.0, 0.0])
    dummy_mesh["Confidence"] = [0.5]  # dummy value  for the hack
    actor = p.add_mesh(dummy_mesh, scalars="Confidence", cmap="viridis", clim=[0, 1], show_scalar_bar=False)
    actor.visibility = False
    sbar_args = dict(
        title="Prediction Confidence",
        title_font_size=20,
        label_font_size=16,
        vertical=False,
        position_x=0.2,
        position_y=0.3,
        height=0.4,
        width=0.6,
        color='black'
    )
    p.add_scalar_bar(**sbar_args)

    p.camera_position = 'xy'
    p.camera.up = (1, 0, 0)
    p.camera.zoom(1)

    p.show()


if __name__ == "__main__":
    active_mesh = "mesh019"
    iterations = 50
    gammas_to_test = [1e-06, 0.002, 0.02, 0.1, 10]
    colors_seg = ["red", "green", "blue", "yellow", "cyan", "purple", "white"]
    run_gamma_analysis(active_mesh=active_mesh, iterations=iterations,
                       gammas_to_test=gammas_to_test, colors_seg=colors_seg)