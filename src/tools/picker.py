import pyvista as pv
from src.geometry.config import MESH_PATH

def main():
    mesh = pv.read(MESH_PATH)
    p = pv.Plotter()
    p.add_text("Click a point and check the index on the terminal, press q to exit.", font_size=12)
    pv_mesh = pv.wrap(mesh)
    p.add_mesh(pv_mesh, color='white', show_edges=True, pickable=True)

    def callback(point, picker):
        if picker.GetPointId() != -1:
            idx = picker.GetPointId()
            print(f"Selected vertex {idx}")
            p.add_mesh(pv.PolyData(point), color='red', point_size=10, render_points_as_spheres=True)

    p.enable_point_picking(callback=callback, show_message=False, use_picker=True)
    p.show()

if __name__ == "__main__":
    main()