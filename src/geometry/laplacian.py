import numpy as np
from scipy import sparse


def build_laplacian(vertices, faces):
    """
    Computes the cotangent laplacian L and mass matrix M.
    """
    vertices = vertices.astype(np.float64)
    n = len(vertices)

    # Compute edge vectors
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Vectors along edges
    e0 = v2 - v1
    e1 = v0 - v2
    e2 = v1 - v0

    normal = np.cross(e0, e1)
    A = np.linalg.norm(normal, axis=1)  # 2 * Area
    A = np.maximum(A, 1e-12)  # clipping for precision

    # Cotangents: cot(theta) = (u dot v) / (2 * Area)
    term0 = np.sum(-e1 * e2, axis=1) / A
    term1 = np.sum(-e0 * e2, axis=1) / A
    term2 = np.sum(-e0 * e1, axis=1) / A

    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    data = np.stack([term0, term1, term2], axis=1).flatten() * 0.5

    W = sparse.coo_matrix((data, (ii, jj)), shape=(n, n))
    W = (W + W.T)


    D = sparse.diags(np.array(W.sum(axis=1)).flatten())  # flatten because diags expects 1D
    L = D - W

    vertex_areas = np.zeros(n)
    triangle_area = A / 2.0  # A was 2*Area
    for i in range(3):
        np.add.at(vertex_areas, faces[:, i], triangle_area / 3.0)  # each vertex gets 1/3 of the total area
    M = sparse.diags(vertex_areas)

    return L, M