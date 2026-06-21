import numpy as np

class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
        self._normalize_geometry()
        self._compute_topology()
        self._compute_area_weights()

    def _normalize_geometry(self):
        # center at origin
        centroid = np.mean(self.vertices, axis=0)
        self.vertices -= centroid

        # scale to unit sphere
        max_dist = np.max(np.linalg.norm(self.vertices, axis=1))
        self.scale_factor = max_dist  # not sure if needed, if we need to revert
        self.vertices /= max_dist

    def _compute_topology(self):
        edges = np.concatenate([
            self.faces[:, [0, 1]],
            self.faces[:, [1, 2]],
            self.faces[:, [2, 0]]
        ], axis=0)
        edges.sort(axis=1)
        self.edges_unique = np.unique(edges, axis=0)

    def _compute_area_weights(self):
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]

        cross = np.cross(v1 - v0, v2 - v0)
        face_areas = 0.5 * np.linalg.norm(cross, axis=1)

        self.a = np.zeros(self.vertices.shape[0])
        for i in range(3):
            np.add.at(self.a, self.faces[:, i], face_areas / 3.0)

        self.a /= np.sum(self.a)

    @property
    def num_vertices(self):
        return self.vertices.shape[0]