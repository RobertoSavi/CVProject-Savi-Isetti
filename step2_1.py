import os
import cv2
import numpy as np
import json
import glob
import re
import matplotlib.pyplot as plt
import pprint
from collections import defaultdict

SCRIPT_DIR = os.getcwd()

IMG_FOLDER = os.path.join(SCRIPT_DIR, "resources", "annotations", "yolo_dataset", "train", "images")
LABELS_FOLDER = os.path.join(SCRIPT_DIR, "resources", "annotations", "yolo_dataset", "train", "labels")
CAMERA_FOLDER = os.path.join(SCRIPT_DIR, "resources", "cameras", "camera_data_with_Rvecs_2ndversion", "camera_data")

RECT_IMG_FOLDER = os.path.join(SCRIPT_DIR, "utils", "rectified_images")
RECT_LABEL_FOLDER = os.path.join(SCRIPT_DIR, "utils", "rectified_labels")
RECT_CAMERA_FOLDER = os.path.join(SCRIPT_DIR, "utils", "rectified_cameras")
RECT_IMG_AND_LABEL_FOLDER = os.path.join(SCRIPT_DIR, "utils", "rectified_images_and_labels")

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


# Load camera calibration matrix and distortion coefficients from a JSON file
def load_calibration(calib_path):
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    mtx = np.array(calib["mtx"], dtype=np.float32)
    dist = np.array(calib["dist"], dtype=np.float32)
    tvecs = np.array(calib["tvecs"], dtype=np.float32).reshape(3, 1)
    rvecs = np.array(calib["rvecs"], dtype=np.float32).reshape(3, 1)
    
    K_rect = None
    if "K_rect" in calib:
        K_rect = np.array(calib["K_rect"], dtype=np.float32)
        
    return mtx, dist, tvecs, rvecs, K_rect

# Parse a YOLO annotation line into bbox and keypoints in pixel coordinates
def parse_annotation_file(label_path, img_w, img_h):
    with open(label_path, 'r') as f:
        label = f.readline().strip()
    parts = label.split()
    class_id = int(parts[0]) # Class ID of the object
    bbox = list(map(float, parts[1:5])) # The four points that define the bounding box
    kp_data = list(map(float, parts[5:])) # Keypoints data in YOLO format

    keypoints = []
    for i in range(0, len(kp_data), 3):
        x = kp_data[i] * img_w
        y = kp_data[i + 1] * img_h
        v = kp_data[i + 2]
        keypoints.append((x, y, v))
    # Also return the class ID, and bounding box alongside the keypoints to reconstruct the full annotation after rectification
    return class_id, bbox, keypoints

# Rectify a list of keypoints given the camera matrices
def undistort_keypoints(keypoints, mtx, dist, new_mtx):
    if not keypoints:
        return []
    # Convert keypoints to shape (N,1,2) for cv2.undistortPoints
    points = np.array([[[x, y]] for x, y, _ in keypoints], dtype=np.float32)
    # Undistort and rectify the points
    undistorted = cv2.undistortPoints(points, mtx, dist, P=new_mtx)
    # Return a flat list with visibility
    return [(int(p[0][0]), int(p[0][1]), keypoints[i][2]) for i, p in enumerate(undistorted)]

# Normalize keypoints to [0, 1] range for saving back to YOLO format
def normalize_keypoints(keypoints, img_w, img_h):
    normed = []
    for x, y, v in keypoints:
        normed.extend([x / img_w, y / img_h, v])
    return normed

# Full process for image and annotation rectification
def rectify_image_and_keypoints(img_path, camera_matrix, label_path, out_img_path, out_label_path, out_img_and_label_path):
    mtx   = np.array(camera_matrix["mtx"],  dtype=np.float32)
    dist  = np.array(camera_matrix["dist"], dtype=np.float32)

    img = cv2.imread(img_path)
    if img is None:
        print("Could not read image:", img_path)
        return

    h, w = img.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha=0)
    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, new_mtx, (w, h), cv2.CV_32FC1)
    rect_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_CUBIC)

    x, y, rw, rh = roi
    K_crop = new_mtx.copy()
    K_crop[0, 2] -= x   # shift cx
    K_crop[1, 2] -= y   # shift cy
    x2 = min(x+rw+1, rect_img.shape[1])
    y2 = min(y+rh+1, rect_img.shape[0])
    rect_img = rect_img[y:y2, x:x2]

    cv2.imwrite(out_img_path, rect_img)

    class_id, bbox, keypoints = parse_annotation_file(label_path, w, h)
    rect_kpts = undistort_keypoints(keypoints, mtx, dist, new_mtx)

    rect_kpts = [(x - roi[0], y - roi[1], v) for x, y, v in rect_kpts]

    rect_kpts_norm = normalize_keypoints(rect_kpts, rw, rh)
    line = f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} " + " ".join([
        f"{x:.6f} {y:.6f} {int(v)}" for x, y, v in zip(rect_kpts_norm[::3], rect_kpts_norm[1::3], rect_kpts_norm[2::3])
    ])

    with open(out_label_path, "w") as f:
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

    cv2.imwrite(os.path.join(out_img_and_label_path, os.path.basename(out_img_path)), vis_img)

    print(f"Saved rectified image and annotation for: {os.path.basename(img_path)}")
    return K_crop

def make_json_friendly(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_json_friendly(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_friendly(v) for v in obj]
    return obj

def main():
    os.makedirs(RECT_IMG_FOLDER, exist_ok=True)
    os.makedirs(RECT_LABEL_FOLDER, exist_ok=True)
    os.makedirs(RECT_CAMERA_FOLDER, exist_ok=True)
    os.makedirs(RECT_IMG_AND_LABEL_FOLDER, exist_ok=True)

    cameras = {}
    for cam_index in CAMERA_INDEXES:
        calib_path = os.path.join(CAMERA_FOLDER, f"cam_{cam_index}", "calib", "camera_calib.json")
        if not os.path.exists(calib_path):
            print(f"Calibration file not found for camera {cam_index}: {calib_path}")
            continue
        mtx, dist, tvecs, rvecs, K_rect = load_calibration(calib_path)
        cameras[cam_index] = {
            "mtx": mtx,
            "dist": dist,
            "tvecs": tvecs,
            "rvecs": rvecs,
            "K_rect": K_rect
        }

    img_paths = glob.glob(os.path.join(IMG_FOLDER, "*.jpg"))

    # Perform rectification for each image and its corresponding label.
    for img_path in img_paths:
        basename = os.path.basename(img_path)
        name_wo_ext = os.path.splitext(basename)[0]
        label_path = os.path.join(LABELS_FOLDER, f"{name_wo_ext}.txt")

        match = re.search(r'out(\d+)_frame_(\d+).*\.jpg$', basename)
        if not match:
            print("Could not extract camera index or frame index from filename:", img_path)
            continue
        cam_index = int(match.group(1))

        out_img_path = os.path.join(RECT_IMG_FOLDER, basename)
        out_label_path = os.path.join(RECT_LABEL_FOLDER, f"{name_wo_ext}.txt")

        K_rect = rectify_image_and_keypoints(
            img_path,
            cameras[cam_index],
            label_path,
            out_img_path,
            out_label_path,
            RECT_IMG_AND_LABEL_FOLDER
        )

        if cameras[cam_index]["K_rect"] is None:
            cameras[cam_index]["K_rect"] = K_rect

    for cam_index, camera in cameras.items():
        cam_path = os.path.join(RECT_CAMERA_FOLDER, f"cam_{cam_index}_calib.json")
        data_to_save = make_json_friendly(camera)

        with open(cam_path, "w") as f:
            json.dump(data_to_save, f, indent=4)
        
            
if __name__ == "__main__":
    main()
