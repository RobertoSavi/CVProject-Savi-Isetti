import os
import cv2
import numpy as np
import json
import glob
import re
import matplotlib.pyplot as plt
import pprint
from collections import defaultdict
import step2_1

TRIANG_FOLDER = os.path.join(os.getcwd(), "utils", "triangulated_points")

def compute_projection_matrix(mtx, tvecs, rvecs):
    K = mtx
    # Convert rvec in R
    R, _ = cv2.Rodrigues(rvecs)
    # Create the matrix [R | t]
    Rt = np.hstack((R, tvecs))
    # Projection matrix
    P = K @ Rt  
    return P
    
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
            _, _, points = step2_1.parse_annotation_file(label_path, step2_1.IMG_WIDTH, step2_1.IMG_HEIGHT)
            all_keypoints.append(points)

        num_points = len(all_keypoints[0])
        triangulated_points = []

        for pt_idx in range(num_points):
            # points_2d: coordinate del pt_idx in tutte le camere
            points_2d = [all_keypoints[cam_idx][pt_idx] for cam_idx in range(len(frame_view))]

            # prendo la matrice di proiezione corrispondente a ogni camera (index 0-3)
            P_list = [projection_matrices[step2_1.CAMERA_INDEXES[cam_idx]] for cam_idx in range(len(frame_view))]

            # triangolazione
            X_3d = triangulate_multi_view(points_2d, P_list)
            triangulated_points.append(X_3d)

        triangulated_points = np.array(triangulated_points)
        output_path = os.path.join(output_folder, f"triangulated_frame_{frame_idx:04d}.txt")
        np.savetxt(output_path, triangulated_points)
        print(f"Saved 3D points for frame {frame_idx} to {output_path}")

def main():
    os.makedirs(TRIANG_FOLDER, exist_ok=True)
    label_paths = glob.glob(os.path.join(step2_1.RECT_LABEL_FOLDER, "*.txt"))
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
        frame_tuple = tuple(frames_dict[frame_index][cam] for cam in step2_1.CAMERA_INDEXES)
        all_frames_views[frame_index] = frame_tuple
        
    projection_matrices = {}
    camera_paths = glob.glob(os.path.join(step2_1.RECT_CAMERA_FOLDER, "*json"))
    camera_paths_sorted = sorted(camera_paths)
    for camera_path in camera_paths_sorted:
        _, _, tvecs, rvecs, K_rect = step2_1.load_calibration(camera_path)
        print(int(os.path.basename(camera_path).split('_')[1]))
        projection_matrices[int(os.path.basename(camera_path).split('_')[1])] = compute_projection_matrix(K_rect, tvecs, rvecs)
        
    # Perform traingulation for all frames
    triangulate_all_frames(projection_matrices, all_frames_views, TRIANG_FOLDER)
            
if __name__ == "__main__":
    main()