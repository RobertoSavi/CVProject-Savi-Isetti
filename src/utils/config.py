import os
import numpy as np

FRAMES_NUMBER = 48
CAMERA_INDEXES = [2, 5, 8, 13]
IMG_HEIGHT = 2160
IMG_WIDTH = 3840
CONNECTIONS = [ 
    (0, 9), (9, 10), (10, 11),   # Head to Hip
    
    (10, 12), (12, 13), (13, 14), # Neck to R-Hand
    (10, 15), (15, 16), (16, 17), # Neck to L-Hand
    
    (0, 1), (1, 2), (2, 3), (3, 4), # Hip to R-Foot
    (0, 5), (5, 6), (6, 7), (7, 8), # Hip to L-Foot
]
LABELS_CONVERTER = {
    0:0, 9:2, 10:4, 11:5, # Head to Hip
    
    12:12, 13:13, 14:15, # Neck to R-Hand
    15:7,  16:8,  17:10, # Neck to L-Hand
    
    1:20, 2:21, 3:22, 4:23, # Hip to R-Foot
    5:16,  6:17,  7:18,  8:19, # Hip to L-Foot
}

MOCAP_FRAMES = 12000
MOCAP_FRAME_RATE = 100

ANNOTATION_ALIGN_FRAME = 22
MOCAP_ALIGN_FRAME = 9965


# Directory paths
SCRIPT_DIR = os.getcwd()

# Input data
IMG_FOLDER = os.path.join(SCRIPT_DIR, "..", "resources", "annotations", "yolo_dataset", "train", "images")
LABELS_FOLDER = os.path.join(SCRIPT_DIR, "..", "resources", "annotations", "yolo_dataset", "train", "labels")
CAMERA_FOLDER = os.path.join(SCRIPT_DIR, "..", "resources", "cameras", "camera_data")
MOCAP_FOLDER = os.path.join(SCRIPT_DIR, "..", "resources", "mocap")

# Output data
RECT_IMG_FOLDER = os.path.join(SCRIPT_DIR, "..", "results", "rectified_images")
RECT_LABEL_FOLDER = os.path.join(SCRIPT_DIR, "..", "results", "rectified_labels")
RECT_CAMERA_FOLDER = os.path.join(SCRIPT_DIR, "..", "results", "rectified_cameras")
RECT_IMG_AND_LABEL_FOLDER = os.path.join(SCRIPT_DIR, "..", "results", "rectified_images_and_labels")

TRIANG_FOLDER = os.path.join(SCRIPT_DIR, "..", "results", "triang_points")
TRIANG_OVERLAYS_FOLDER = os.path.join(SCRIPT_DIR, "..", "results", "triang_overlays")
TRIANG_ERROR_PLOTS_FOLDER = os.path.join(SCRIPT_DIR, "..", "results", "triang_vs_annotation_error_plots")
TRIANG_3D_FOLDER = os.path.join(SCRIPT_DIR, "..", "results", "triang_3d")
MOCAP_OVERLAYS_FOLDER = os.path.join(SCRIPT_DIR, "..", "results", "mocap_overlays")
MOCAP_ERROR_PLOTS_FOLDER = os.path.join(SCRIPT_DIR, "..", "results", "triang_vs_mocap_error_plots")
MOCAP_TRIANG_3D_FOLDER = os.path.join(SCRIPT_DIR, "..", "results", "mocap_triang_3d")


def make_json_friendly(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_json_friendly(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_friendly(v) for v in obj]
    return obj