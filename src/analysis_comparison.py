import numpy as np
import pyvista as pv
from scipy import stats
from scipy.spatial import KDTree
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra

from src.geometry.config import MESH_PATH, MESH_CONFIGS
from src.geometry.mesh import Mesh
from src.solvers.heat import HeatKernel
from src.solvers.sinkhorn import SinkhornSolver

def compute_geodesic_ground_truth(mesh, seeds_dict):
    """
    computes the geodesic ground truth with a simple dijkstra
    """
    N = mesh.num_vertices
    adj = lil_matrix((N, N))
    edges = mesh.edges_unique
    v1 = mesh.vertices[edges[:, 0]]
    v2 = mesh.vertices[edges[:, 1]]
    lengths = np.linalg.norm(v1 - v2, axis=1)
    adj[edges[:, 0], edges[:, 1]] = lengths
    adj[edges[:, 1], edges[:, 0]] = lengths
    adj_csr = adj.tocsr()

    num_classes = len(seeds_dict)
    dists = np.zeros((N, num_classes))
    sorted_seeds = sorted(seeds_dict.items(), key=lambda x: x[1])
    for v_idx, cls in sorted_seeds:
        dists[:, cls] = dijkstra(csgraph=adj_csr, directed=False, indices=v_idx)
    return dists


def calculate_violation_score(labels, dists_to_seeds):
    N = len(labels)
    min_dists = np.min(dists_to_seeds, axis=1)
    assigned_dists = dists_to_seeds[np.arange(N), labels]
    return np.maximum(assigned_dists - min_dists, 0.0)


def solve_sinkhorn(mesh, seeds, iterations):
    kernel = HeatKernel(mesh, gamma=gamma)
    solver = SinkhornSolver(mesh, kernel)
    plan = solver.solve(seeds, len(seeds), iterations=iterations)
    row_sums = np.sum(plan, axis=1, keepdims=True)
    plan /= (row_sums + 1e-20)
    return np.argmax(plan, axis=1)


def solve_euclidean(mesh, seeds):
    seed_positions = [mesh.vertices[v] for v in seeds.keys()]
    seed_labels = list(seeds.values())
    tree = KDTree(seed_positions)
    _, indices = tree.query(mesh.vertices)
    return np.array([seed_labels[i] for i in indices])


def vertices_to_faces_majority_vote(vertex_labels, faces):
    l0 = vertex_labels[faces[:, 0]]
    l1 = vertex_labels[faces[:, 1]]
    l2 = vertex_labels[faces[:, 2]]
    modes, _ = stats.mode(np.stack([l0, l1, l2]), axis=0, keepdims=True)
    return modes[0]

def analysis_comparison(active_mesh, iterations, gamma, colors_seg):
    print("Running comparison analysis")
    # duplication of the main script basically
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

    # solve our and the euclidean approaches
    dists = compute_geodesic_ground_truth(mesh, seeds)
    lbl_geo = solve_sinkhorn(mesh, seeds, iterations)
    lbl_euc = solve_euclidean(mesh, seeds)

    # violation scores for both
    vio_geo = calculate_violation_score(lbl_geo, dists)
    vio_euc = calculate_violation_score(lbl_euc, dists)

    global_max_error = np.max(vio_euc)  # to have colorbar same for both

    print(f"Violation Scores:")
    print(f"Ours (Max): {np.max(vio_geo):.4f}")
    print(f"Euclidean (Max): {global_max_error:.4f}")

    p = pv.Plotter(
        shape=(3, 2),
        row_weights=[1, 1, 0.2],
        groups=[(2, np.s_[:])],  # pyvista hack from https://github.com/pyvista/pyvista/discussions/4542
        border=False
    )
    p.link_views()

    base_m = pv.wrap(mesh_data).copy(deep=True)
    base_m.points = mesh.vertices

    # Row 0: labels
    # Top left: naive
    p.subplot(0, 0)
    p.add_text("1. Naive Euclidean", font_size=10)
    m1 = base_m.copy(deep=True)
    face_lbl = vertices_to_faces_majority_vote(lbl_euc, mesh.faces)
    m1.cell_data["Labels"] = face_lbl
    p.add_mesh(m1, scalars="Labels", cmap=colors_seg, clim=[0, 6], show_scalar_bar=False)

    # Top right: ours
    p.subplot(0, 1)
    p.add_text(f"2. Ours (Gamma={gamma})", font_size=10)
    m2 = base_m.copy(deep=True)
    face_lbl_geo = vertices_to_faces_majority_vote(lbl_geo, mesh.faces)
    m2.cell_data["Labels"] = face_lbl_geo
    p.add_mesh(m2, scalars="Labels", cmap=colors_seg, clim=[0, 6], show_scalar_bar=False)

    # Row 1: violation
    # Bottom left: naive error
    p.subplot(1, 0)
    p.add_text(f"3. Euclidean Violation\n(Max: {global_max_error:.2f})", font_size=10)
    m3 = base_m.copy(deep=True)
    m3["Violation"] = vio_euc
    p.add_mesh(m3, scalars="Violation", cmap="inferno", clim=[0, global_max_error], show_scalar_bar=False)

    # Bottom right: our error
    p.subplot(1, 1)
    p.add_text(f"4. Our Violation\n(Max: {np.max(vio_geo):.2f})", font_size=10)
    m4 = base_m.copy(deep=True)
    m4["Violation"] = vio_geo
    p.add_mesh(m4, scalars="Violation", cmap="inferno", clim=[0, global_max_error], show_scalar_bar=False)

    # colorbar with dummy hack
    p.subplot(2, 0)
    dummy_mesh = pv.PolyData([0.0, 0.0, 0.0])
    dummy_mesh["Violation"] = [0.0]
    actor = p.add_mesh(dummy_mesh, scalars="Violation", cmap="inferno", clim=[0, global_max_error],
                       show_scalar_bar=False)
    actor.visibility = False
    sbar_args = dict(
        title="Geodesic Violation Score (Error Magnitude)",
        title_font_size=20,
        label_font_size=16,
        vertical=False,
        position_x=0.2,
        position_y=0.1,
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
    iterations = 100
    gamma = 0.02
    colors_seg = ["red", "green", "blue", "yellow", "cyan", "purple", "white"]
    analysis_comparison(active_mesh=active_mesh, iterations=iterations, gamma=gamma, colors_seg=colors_seg)