# Perform frames and annotations rectification
# TODO: Triangulation
import os
import cv2
import numpy as np
import json
import glob
import re

# Load camera calibration matrix and distortion coefficients from a JSON file.
def load_calibration(calib_path):
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    mtx = np.array(calib["mtx"], dtype=np.float32)
    dist = np.array(calib["dist"], dtype=np.float32)
    return mtx, dist

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

# Normalize keypoints to [0, 1] range for saving back to YOLO format.
def normalize_keypoints(keypoints, img_w, img_h):
    normed = []
    for x, y, v in keypoints:
        normed.extend([x / img_w, y / img_h, v])
    return normed

# Full process for image and annotation rectification.
def rectify_image_and_keypoints(image_path, calib_path, label_path, output_img_path, output_label_path, output_img_and_label_path):
    mtx, dist = load_calibration(calib_path)

    img = cv2.imread(image_path)
    if img is None:
        print("Could not read image:", image_path)
        return

    h, w = img.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha=0)
    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, new_mtx, (w, h), cv2.CV_32FC1)
    rect_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_CUBIC)

    x, y, rw, rh = roi
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

    # Save visualization with keypoints and connections
    annotated_folder = output_img_and_label_path
    os.makedirs(annotated_folder, exist_ok=True)
    vis_img = rect_img.copy()

    # Connections between keypoints (index pairs)
    connections = [ 
        (10, 11), (9, 10), (0, 9), # Head to Hip
        (10, 12), (12, 13), (13, 14), # Neck to R-Hand
        (10, 15), (15, 16), (16, 17), # Neck to L-Hand
        (0, 1), (1, 2), (2, 3), (3, 4), # Hip to R-Foot
        (0, 5), (5, 6), (6, 7), (7, 8), # Hip to L-Foot
    ]

    # Draw joints in green
    for idx, (x, y, v) in enumerate(rect_kpts):
        if v > 0:
            cv2.circle(vis_img, (int(x), int(y)), 3, (0, 255, 0), -1)

    # Draw connections in red
    for a, b in connections:
        if rect_kpts[a][2] > 0 and rect_kpts[b][2] > 0:
            pt1 = (int(rect_kpts[a][0]), int(rect_kpts[a][1]))
            pt2 = (int(rect_kpts[b][0]), int(rect_kpts[b][1]))
            cv2.line(vis_img, pt1, pt2, (0, 0, 255), 1)

    cv2.imwrite(os.path.join(annotated_folder, os.path.basename(output_img_path)), vis_img)

    print(f"Saved rectified image and annotation for: {os.path.basename(image_path)}")

# Main execution loop: processes all images in a folder.
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, "resources", "annotations", "yolo_dataset", "train", "images")
    label_folder = os.path.join(script_dir, "resources", "annotations", "yolo_dataset", "train", "labels")
    output_img_folder = os.path.join(script_dir, "utils", "rectified_images")
    output_label_folder = os.path.join(script_dir, "utils", "rectified_labels")
    output_img_and_label_folder = os.path.join(script_dir, "utils", "rectified_images_and_labels")

    os.makedirs(output_img_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)
    os.makedirs(output_img_and_label_folder, exist_ok=True)

    image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))

    for image_path in image_paths:
        basename = os.path.basename(image_path)
        name_wo_ext = os.path.splitext(basename)[0]
        label_path = os.path.join(label_folder, f"{name_wo_ext}.txt")

        match = re.search(r'out(\d+).*\.jpg$', basename)
        if match:
            cam_index = match.group(1)
            calib_path = os.path.join(script_dir, "resources", "cameras", 
                                    "camera_data_with_Rvecs_2ndversion/camera_data", 
                                    f"cam_{cam_index}", "calib", "camera_calib.json")
        else:
            print("Could not extract camera index from filename:", image_path)
            continue

        output_img_path = os.path.join(output_img_folder, basename)
        output_label_path = os.path.join(output_label_folder, f"{name_wo_ext}.txt")

        rectify_image_and_keypoints(image_path, calib_path, label_path, output_img_path, output_label_path, output_img_and_label_folder)

if __name__ == "__main__":
    main()
