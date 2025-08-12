# Perform frames and annotations rectification
# TODO: Triangulation
import os
import cv2
import numpy as np
import json
import glob
import re
import matplotlib.pyplot as plt
import pprint

from collections import defaultdict
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

# Load camera calibration matrix and distortion coefficients from a JSON file.
def load_calibration(calib_path):
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    mtx = np.array(calib["mtx"], dtype=np.float32)
    dist = np.array(calib["dist"], dtype=np.float32)
    tvecs = np.array(calib["tvecs"], dtype=np.float32).reshape(3, 1)
    rvecs = np.array(calib["rvecs"], dtype=np.float32).reshape(3, 1)
    return mtx, dist, tvecs, rvecs

# Parse a YOLO annotation line into bbox and keypoints in pixel coordinates.
def parse_annotation_file(label_path, img_w, img_h):
    with open(label_path, 'r') as f:
        line = f.readline().strip()
    parts = line.split()
    class_id = int(parts[0])
    bbox = list(map(float, parts[1:5]))
    kp_data = list(map(float, parts[5:]))

    keypoints = []
    for i in range(0, len(kp_data), 3):
        x = kp_data[i] * img_w
        y = kp_data[i + 1] * img_h
        v = kp_data[i + 2]
        keypoints.append((x, y, v))

    return class_id, bbox, keypoints

# Undistort and rectify a list of keypoints given the camera matrices.
def undistort_keypoints(keypoints, mtx, dist, new_mtx):
    if not keypoints:
        return []
    # Convert keypoints to shape (N,1,2) for cv2.undistortPoints
    points = np.array([[[x, y]] for x, y, _ in keypoints], dtype=np.float32)
    # Undistort and rectify the points
    undistorted = cv2.undistortPoints(points, mtx, dist, P=new_mtx)
    # Return a flat list with visibility
    return [(int(p[0][0]), int(p[0][1]), keypoints[i][2]) for i, p in enumerate(undistorted)]

# Normalize keypoints to [0, 1] range for saving back to YOLO format.
def normalize_keypoints(keypoints, img_w, img_h):
    normed = []
    for x, y, v in keypoints:
        normed.extend([x / img_w, y / img_h, v])
    return normed

# Full process for image and annotation rectification.
def rectify_image_and_keypoints(image_path, camera_matrix, label_path, output_img_path, output_label_path, output_img_and_label_path):
    mtx, dist, _, _ = camera_matrix["calibration"]

    img = cv2.imread(image_path)
    if img is None:
        print("Could not read image:", image_path)
        return

    h, w = img.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha=0)
    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, new_mtx, (w, h), cv2.CV_32FC1)
    rect_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_CUBIC)

    x, y, rw, rh = roi
    K_crop = new_mtx.copy()
    K_crop[0, 2] -= x   # shift cx
    K_crop[1, 2] -= y   # shift cy
    
    rect_img = rect_img[y:y+rh, x:x+rw]

    cv2.imwrite(output_img_path, rect_img)

    class_id, bbox, keypoints = parse_annotation_file(label_path, w, h)
    rect_kpts = undistort_keypoints(keypoints, mtx, dist, new_mtx)

    rect_kpts = [(x - roi[0], y - roi[1], v) for x, y, v in rect_kpts]

    rect_kpts_norm = normalize_keypoints(rect_kpts, rw, rh)
    line = f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} " + " ".join([
        f"{x:.6f} {y:.6f} {int(v)}" for x, y, v in zip(rect_kpts_norm[::3], rect_kpts_norm[1::3], rect_kpts_norm[2::3])
    ])

    with open(output_label_path, "w") as f:
        f.write(line + "\n")

    vis_img = rect_img.copy()

    # Draw joints in green
    for idx, (x, y, v) in enumerate(rect_kpts):
        if v > 0:
            cv2.circle(vis_img, (int(x), int(y)), 3, (0, 255, 0), -1)

    # Draw connections in red
    for a, b in CONNECTIONS:
        if rect_kpts[a][2] > 0 and rect_kpts[b][2] > 0:
            pt1 = (int(rect_kpts[a][0]), int(rect_kpts[a][1]))
            pt2 = (int(rect_kpts[b][0]), int(rect_kpts[b][1]))
            cv2.line(vis_img, pt1, pt2, (0, 0, 255), 1)

    cv2.imwrite(os.path.join(output_img_and_label_path, os.path.basename(output_img_path)), vis_img)

    print(f"Saved rectified image and annotation for: {os.path.basename(image_path)}")
    return K_crop

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
            _, _, points = parse_annotation_file(label_path, IMG_WIDTH, IMG_HEIGHT)
            all_keypoints.append(points)

        num_points = len(all_keypoints[0])
        triangulated_points = []

        for pt_idx in range(num_points):
            # points_2d: coordinate del pt_idx in tutte le camere
            points_2d = [all_keypoints[cam_idx][pt_idx] for cam_idx in range(len(frame_view))]

            # prendo la matrice di proiezione corrispondente a ogni camera (index 0-3)
            P_list = [projection_matrices[CAMERA_INDEXES[cam_idx]] for cam_idx in range(len(frame_view))]

            # triangolazione
            X_3d = triangulate_multi_view(points_2d, P_list)
            triangulated_points.append(X_3d)

        triangulated_points = np.array(triangulated_points)
        output_path = os.path.join(output_folder, f"triangulated_frame_{frame_idx:04d}.txt")
        np.savetxt(output_path, triangulated_points)
        print(f"Saved 3D points for frame {frame_idx} to {output_path}")
        
def plot_3d_points(points_3d, connections=None):
    """
    points_3d: numpy array Nx3 con coordinate 3D (X, Y, Z)
    connections: lista di tuple (start_idx, end_idx) indicanti le linee da disegnare tra punti
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    xs = points_3d[:, 0]
    ys = points_3d[:, 1]
    zs = points_3d[:, 2]
    
    ax.scatter(xs, ys, zs, c='r', marker='o')

    # Disegna le connessioni se specificate
    if connections is not None:
        for (start_idx, end_idx) in connections:
            x = [points_3d[start_idx, 0], points_3d[end_idx, 0]]
            y = [points_3d[start_idx, 1], points_3d[end_idx, 1]]
            z = [points_3d[start_idx, 2], points_3d[end_idx, 2]]
            ax.plot(x, y, z, c='b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Triangulated 3D Points')
    plt.show()   
    
def draw_points_on_rectified(image_path, camera_matrix, points_3d, output_triangulation_draws, color=(0,255,0), r=3):
    img = cv2.imread(image_path)
    if img is None:
        print("Could not read image:", image_path)
        return
    K_crop =  camera_matrix["new_mtx"] if "new_mtx" in camera_matrix else None
    _, _, tvecs, rvecs = camera_matrix["calibration"]
    R, _ = cv2.Rodrigues(rvecs)  # 3x3 rotation
    
    points_3d = np.asarray(points_3d, dtype=np.float64)
    X_cam = (R @ points_3d.T + tvecs).T                      # (N,3)
    Z = X_cam[:, 2]
    valid = Z > 0                                        # in front of camera
    X_cam = X_cam[valid]
    if len(X_cam) == 0:
        return img

    uv = (K_crop @ X_cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]        
        

    out = img.copy()
    for (u, v) in uv:
        cv2.circle(out, (int(round(u)), int(round(v))), r, color, -1)
    cv2.imwrite(os.path.join(output_triangulation_draws, os.path.basename(image_path)), out)# dehomogenize
        
# Main execution loop: processes all images in a folder.
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_folder = os.path.join(script_dir, "resources", "annotations", "yolo_dataset", "train", "images")
    labels_folder = os.path.join(script_dir, "resources", "annotations", "yolo_dataset", "train", "labels")
    camera_matrices_folder = os.path.join(script_dir, "resources", "cameras", "camera_data_with_Rvecs_2ndversion", "camera_data")
    output_img_folder = os.path.join(script_dir, "utils", "rectified_images")
    output_label_folder = os.path.join(script_dir, "utils", "rectified_labels")
    output_img_and_label_folder = os.path.join(script_dir, "utils", "rectified_images_and_labels")
    output_triangulation_folder = os.path.join(script_dir, "utils", "triangulated_points")
    output_triangulation_folder_draws = os.path.join(script_dir, "utils", "triangulated_points_draws")

    os.makedirs(output_img_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)
    os.makedirs(output_img_and_label_folder, exist_ok=True)
    os.makedirs(output_triangulation_folder, exist_ok=True)
    
    
    cameras={}
    for cam_index in CAMERA_INDEXES:
        calib_path = os.path.join(camera_matrices_folder, f"cam_{cam_index}", "calib", "camera_calib.json")
        if not os.path.exists(calib_path):
            print(f"Calibration file not found for camera {cam_index}: {calib_path}")
            continue
        cameras[cam_index] = {
            "calibration": load_calibration(calib_path),
            "new_mtx": None
    }
    
    # Perform rectification for each image and its corresponding label.
    image_paths = glob.glob(os.path.join(images_folder, "*.jpg"))
    for image_path in image_paths:
        basename = os.path.basename(image_path)
        name_wo_ext = os.path.splitext(basename)[0]
        label_path = os.path.join(labels_folder, f"{name_wo_ext}.txt")

        match = re.search(r'out(\d+)_frame_(\d+).*\.jpg$', basename)
        if match:
            cam_index = int(match.group(1))
            frame_index = match.group(2)
        else:
            print("Could not extract camera index from filename:", image_path)
            continue
        #rectify_image(image_path, cameras[cam_index], output_img_folder)
        output_img_path = os.path.join(output_img_folder, basename)
        output_label_path = os.path.join(output_label_folder, f"{name_wo_ext}.txt")
        new_mtx = rectify_image_and_keypoints(image_path, cameras[cam_index], label_path, output_img_path, output_label_path, output_img_and_label_folder)
        if cameras[cam_index]["new_mtx"] is None:
            cameras[cam_index]["new_mtx"] = new_mtx
        

    projection_matrices = {}
    for cam_index, data in cameras.items():
        _, _, tvecs, rvecs = data["calibration"]
        new_mtx = data["new_mtx"]
        projection_matrices[cam_index] = compute_projection_matrix(new_mtx, tvecs, rvecs)
        
        
    label_paths = glob.glob(os.path.join(labels_folder, "*.txt"))
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
        frame_tuple = tuple(frames_dict[frame_index][cam] for cam in CAMERA_INDEXES)
        all_frames_views[frame_index] = frame_tuple
        
    # Perform traingulation for all frames
    triangulate_all_frames(projection_matrices, all_frames_views, output_triangulation_folder)
        
    for triangulated_file in sorted(glob.glob(os.path.join(output_triangulation_folder, "*.txt"))):
        points_3d = np.loadtxt(triangulated_file)
        print(f"Plotting 3D points for {triangulated_file}")
        connections = CONNECTIONS
        plot_3d_points(points_3d, connections=connections)
        
    image_paths = glob.glob(os.path.join(output_img_folder, "*.jpg"))
    
    for image_path in image_paths:
        basename = os.path.basename(image_path)
        name_wo_ext = os.path.splitext(basename)[0]
        label_path = os.path.join(labels_folder, f"{name_wo_ext}.txt")

        match = re.search(r'out(\d+)_frame_(\d+).*\.jpg$', basename)
        if match:
            cam_index = int(match.group(1))
            frame_index = int(match.group(2))

        else:
            print("Could not extract camera index from filename:", image_path)
            continue
        triangulation_file = os.path.join(output_triangulation_folder, f"triangulated_frame_{frame_index:04d}.txt")
        points_3d = np.loadtxt(path, dtype=np.float64)  # shape (N, 3)
        draw_points_on_rectified(image_path, cameras[cam_index], points_3d, output_triangulation_folder_draws, color=(0,255,0), r=3)
        
        
    
        
        
if __name__ == "__main__":
    main()
