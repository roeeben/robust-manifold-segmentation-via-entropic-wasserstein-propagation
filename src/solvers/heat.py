import scipy.sparse as sp
from scipy.sparse.linalg import factorized
from src.geometry.laplacian import build_laplacian


class HeatKernel:
    def __init__(self, mesh, gamma=0.05):
        """
        Implements the heat kernel H_t and solves (D_a + tL)w = a * v before equation 9 of the base paper,
        according to t = gamma / 2.
        """
        print(f"Initializing heat kernel with gamma={gamma}")
        self.mesh = mesh
        L, _ = build_laplacian(mesh.vertices, mesh.faces)
        D_a = sp.diags(mesh.a)
        self.t = gamma / 2.0  # equation 7 of the base paper

        # pre-fractorize
        system_matrix = D_a + self.t * L
        self.solve_system = factorized(system_matrix)

    def apply(self, v):
        """
        Applies the operator H_t to a vector v.
        """
        rhs = self.mesh.a * v
        return self.solve_system(rhs)