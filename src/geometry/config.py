import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "..", "data")
MESH_PATH = os.path.join(DATA_DIR, "mesh019.ply")

# mesh005 and 019 are from the SCAPE dataset
MESH_CONFIGS = {
    "mesh005": {  # easier to segment
        "filename": "mesh005.ply", 
        "anchors": {0: 12471, 3: 9009, 4: 8257, 5: 2737, 6: 2645, 2: 8237}
    },
    "mesh019": {  # harder to segment
        "filename": "mesh019.ply",
        "anchors": {0: 12476, 3: 8649, 4: 8776, 5: 2737, 6: 2645, 2: 8630}
    }
}