import numpy as np

class SinkhornSolver:
    def __init__(self, mesh, kernel):
        self.mesh = mesh
        self.kernel = kernel

    def solve(self, seeds, num_classes, iterations=50):
        """
        Computes the Optimal Transport plan from the set of seeds to the mesh.
        Matches Proposition 1 from Solomon et al. (2015).
        """
        N = self.mesh.num_vertices

        # mu_0 (source distribution) is a sum of diracs
        mu_0 = np.full(N, 1e-10)
        for v_idx, cls in seeds.items():
            mu_0[v_idx] += 1.0
        mu_0 /= np.sum(mu_0)  # norm to 1

        # mu_1 (target distribution) is uniform
        mu_1 = np.ones(N)
        mu_1 /= np.sum(mu_1 * self.mesh.a)  # norm by area

        v = np.ones(N)
        w = np.ones(N)

        print(f"Running Sinkhorn for {iterations} iterations")
        for i in range(iterations):
            H_aw = self.kernel.apply(w)  # diffuse a * w
            H_aw = np.maximum(H_aw, 1e-100)  # stability
            v = mu_0 / H_aw

            H_av = self.kernel.apply(v)  # diffuse a * v
            H_av = np.maximum(H_av, 1e-100)  # stability
            w = mu_1 / H_av

        # build pi column by column
        transport_plan = np.zeros((N, num_classes))
        for v_idx_seed, cls in seeds.items():
            dirac = np.zeros(N)
            dirac[v_idx_seed] = 1.0 / self.mesh.a[v_idx_seed]

            heat_from_seed = self.kernel.apply(dirac)
            mass_from_class = v[v_idx_seed] * heat_from_seed * w  # proposition 1

            transport_plan[:, cls] += mass_from_class

        return transport_plan