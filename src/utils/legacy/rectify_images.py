import cv2
import numpy as np
import json
import os
import glob
import re

def load_calibration(calib_path):
    # Load the camera calibration parameters from a JSON file.
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    mtx = np.array(calib["mtx"], dtype=np.float32)
    dist = np.array(calib["dist"], dtype=np.float32)
    return mtx, dist

def rectify_image(image_path, calib_path, output_path):
    mtx, dist = load_calibration(calib_path)

    img = cv2.imread(image_path)
    if img is None:
        print("Could not read image:", image_path)
        return

    height, width = img.shape[:2]

    # Calculate new camera matrix with alpha=0 to remove black borders
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), alpha=0)

    # Calculate the undistortion and rectification transformation map
    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, new_mtx, (width, height), cv2.CV_32FC1)

    # Remap and crop
    undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_CUBIC)
    x, y, rw, rh = roi
    cropped = undistorted[y:y+rh, x:x+rw]

    cv2.imwrite(output_path, cropped)
    print(f"Saved rectified image to: {output_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, "..", "resources", "annotations", "yolo_dataset", "train", "images")
    image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))  # Estensione regolabile
    print(f"Found {len(image_paths)} images to process.")

    output_folder = os.path.join(script_dir, "rectified_images2")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for image_path in image_paths:
        basename = os.path.basename(image_path)
        match = re.search(r'out(\d+).*\.jpg$', basename)
        if match:
            cam_index = match.group(1)
            calib_path = os.path.join(script_dir, "..", "resources", "cameras", 
                                      "camera_data_with_Rvecs_2ndversion/camera_data", 
                                      f"cam_{cam_index}", "calib", "camera_calib.json")
        else:
            print("Could not extract camera index from filename:", image_path)
            continue

        output_path = os.path.join(output_folder, basename)
        print(f"Processing {image_path} using calibration file {calib_path}...")
        rectify_image(image_path, calib_path, output_path)

if __name__ == "__main__":
    main()