import os
import cv2
import numpy as np
import json
import glob
import re
import matplotlib.pyplot as plt
import pprint
from collections import defaultdict
import utils.config as config
import utils.calibration_utils as calib
import utils.annotation_utils as annot
    
def triangulate_multi_view(points_2d, projection_matrices):
    A = []
    for (x, y, _), P in zip(points_2d, projection_matrices):
        A.append(x * P[2] - P[0])  # u-eq: x * P3 - P1
        A.append(y * P[2] - P[1])  # v-eq: y * P3 - P2
    A = np.asarray(A, dtype=np.float64)

    _, _, Vt = np.linalg.svd(A, full_matrices=False)  # solve homogeneous LS
    Xh = Vt[-1]                                       # last right-singular vector
    Xh /= Xh[-1]                                      # dehomogenize (W=1)
    return Xh[:3]                                     # return (X,Y,Z)


def triangulate_all_frames(projection_matrices, all_frames_views, output_folder):
    for frame_idx, frame_view in sorted(all_frames_views.items()):
        all_keypoints = []
        for label_path in frame_view:
            _, _, points = annot.parse_annotation_file(label_path, config.IMG_WIDTH, config.IMG_HEIGHT)
            all_keypoints.append(points)

        num_points = len(all_keypoints[0])
        triangulated_points = []

        for pt_idx in range(num_points):
            # points_2d: coordinate del pt_idx in tutte le camere
            points_2d = [all_keypoints[cam_idx][pt_idx] for cam_idx in range(len(frame_view))]

            # prendo la matrice di proiezione corrispondente a ogni camera (index 0-3)
            P_list = [projection_matrices[config.CAMERA_INDEXES[cam_idx]] for cam_idx in range(len(frame_view))]

            # triangolazione
            X_3d = triangulate_multi_view(points_2d, P_list)
            triangulated_points.append(X_3d)

        triangulated_points = np.array(triangulated_points)
        output_path = os.path.join(output_folder, f"triangulated_frame_{frame_idx:04d}.txt")
        np.savetxt(output_path, triangulated_points)
        print(f"Saved 3D points for frame {frame_idx} to {output_path}")

def main():
    os.makedirs(config.TRIANG_FOLDER, exist_ok=True)
    label_paths = glob.glob(os.path.join(config.RECT_LABEL_FOLDER, "*.txt"))
    label_paths_sorted = sorted(label_paths)
    
    frames_dict = defaultdict(dict)
    for path in label_paths_sorted:
        basename = os.path.basename(path)
        match = re.search(r'out(\d+)_frame_(\d+)', basename)
        if match:
            cam_index = int(match.group(1))
            frame_index = int(match.group(2))
            frames_dict[frame_index][cam_index] = path

    all_frames_views = {}
    for frame_index in sorted(frames_dict.keys()):
        frame_tuple = tuple(frames_dict[frame_index][cam] for cam in config.CAMERA_INDEXES)
        all_frames_views[frame_index] = frame_tuple
        
    projection_matrices = {}

    for cam_idx in config.CAMERA_INDEXES:
        cam_path = os.path.join(config.RECT_CAMERA_FOLDER, f"cam_{cam_idx}_calib.json")
        if not os.path.exists(cam_path):
            print(f"Missing calib for cam {cam_idx}: {cam_path}")
            continue

        # Load calib & compute P once
        mtx, dist, tvecs, rvecs, K_rect, _ = calib.load_calibration(cam_path)
        P = calib.compute_projection_matrix(K_rect, tvecs, rvecs)
        projection_matrices[cam_idx] = P

        # Read existing JSON (preserve other fields)
        try:
            with open(cam_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

        newP = np.asarray(P, dtype=float).tolist()

        # Only update if changed (avoids touching timestamps needlessly)
        if data.get("P") != newP:
            data["P"] = newP
            tmp = cam_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, cam_path)
            print(f"Updated P for cam {cam_idx}")
        else:
            print(f"P already up-to-date for cam {cam_idx}")
        
    # Perform traingulation for all frames
    triangulate_all_frames(projection_matrices, all_frames_views, config.TRIANG_FOLDER)
            
if __name__ == "__main__":
    main()