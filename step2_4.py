import os
import re
import glob
import json
import numpy as np
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ==== CONFIG ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_DIR = os.path.join(BASE_DIR, "utils", "rectified_labels")
IMAGES_DIR = os.path.join(BASE_DIR, "utils", "rectified_images")  # rectified images from step2
TRIANG_DIR = os.path.join(BASE_DIR, "utils", "triangulated_points")
CAMERA_MATRICES_DIR = os.path.join(BASE_DIR, "resources", "cameras", "camera_data_with_Rvecs_2ndversion", "camera_data")
OUTPUT_OVERLAYS_DIR = os.path.join(BASE_DIR, "utils", "reprojected_overlays")
OUTPUT_PLOTS_DIR = os.path.join(BASE_DIR, "utils", "error_plots")

CAMERA_INDEXES = [2, 5, 8, 13]
IMG_WIDTH = 3840
IMG_HEIGHT = 2160
CONNECTIONS = [
    (0, 9), (9, 10), (10, 11),
    (10, 12), (12, 13), (13, 14),
    (10, 15), (15, 16), (16, 17),
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
]

os.makedirs(OUTPUT_OVERLAYS_DIR, exist_ok=True)
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

# ==== FUNCTIONS ====
def load_calibration(calib_path):
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    mtx = np.array(calib["mtx"], dtype=np.float32)
    dist = np.array(calib["dist"], dtype=np.float32)
    tvecs = np.array(calib["tvecs"], dtype=np.float32).reshape(3, 1)
    rvecs = np.array(calib["rvecs"], dtype=np.float32).reshape(3, 1)
    return mtx, dist, tvecs, rvecs

def parse_annotation_file(label_path, img_w, img_h):
    with open(label_path, 'r') as f:
        line = f.readline().strip()
    parts = line.split()
    kp_data = list(map(float, parts[5:]))
    keypoints = []
    for i in range(0, len(kp_data), 3):
        x = kp_data[i] * img_w
        y = kp_data[i + 1] * img_h
        v = kp_data[i + 2]
        keypoints.append((x, y, v))
    return keypoints

def reproject_points(points_3d, mtx, dist, rvecs, tvecs):
    points_2d, _ = cv2.projectPoints(points_3d, rvecs, tvecs, mtx, dist)
    return points_2d.reshape(-1, 2)

def draw_skeleton_overlay(img, gt_keypoints, reproj_keypoints):
    out = img.copy()

    # Draw GT in green
    for a, b in CONNECTIONS:
        if gt_keypoints[a][2] > 0 and gt_keypoints[b][2] > 0:
            cv2.line(out,
                     (int(round(gt_keypoints[a][0])), int(round(gt_keypoints[a][1]))),
                     (int(round(gt_keypoints[b][0])), int(round(gt_keypoints[b][1]))),
                     (0, 255, 0), 1)
    for x, y, v in gt_keypoints:
        if v > 0:
            cv2.circle(out, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1)

    # Draw Reprojection in blue
    for a, b in CONNECTIONS:
        if reproj_keypoints[a][2] > 0 and reproj_keypoints[b][2] > 0:
            cv2.line(out,
                     (int(round(reproj_keypoints[a][0])), int(round(reproj_keypoints[a][1]))),
                     (int(round(reproj_keypoints[b][0])), int(round(reproj_keypoints[b][1]))),
                     (255, 0, 0), 1)
    for x, y, v in reproj_keypoints:
        if v > 0:
            cv2.circle(out, (int(round(x)), int(round(y))), 3, (255, 0, 0), -1)

    return out


def plot_errors(errors_list, output_dir):
    """
    errors_list: list of dicts with keys: 'frame', 'camera', 'mpjpe', 'mse'
    """
    frames = [e['frame'] for e in errors_list]
    cameras = [e['camera'] for e in errors_list]
    mpjpes = [e['mpjpe'] for e in errors_list]
    mses = [e['mse'] for e in errors_list]

    colors = {2: 'red', 5: 'blue', 8: 'green', 13: 'orange'}

    def apply_y_ticks(ax, data, zoom=False, is_mse=False):
        """Set nice y-axis ticks depending on zoom/full and metric type"""
        max_val = np.max(data)
        if zoom:
            max_val = np.percentile(data, 95)

        # heuristic step size depending on metric
        if is_mse:
            step = max(50, round(max_val / 15, -1))   # ~15 ticks, rounded to nearest 10
        else:
            step = max(5, round(max_val / 15, -1))    # ~15 ticks, rounded to nearest 5

        ax.yaxis.set_major_locator(ticker.MultipleLocator(step))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        if is_mse:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        else:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

        ax.grid(True, which="both", linestyle="--", alpha=0.6)

    # ========== MPJPE PLOT (FULL) ==========
    plt.figure(figsize=(12, 5))
    for cam in sorted(set(cameras)):
        cam_frames = [f for f, c in zip(frames, cameras) if c == cam]
        cam_vals = [m for m, c in zip(mpjpes, cameras) if c == cam]
        plt.plot(cam_frames, cam_vals, 'o-', color=colors.get(cam, 'black'), label=f'Camera {cam}')

    plt.xticks(frames, [f"{frame}" for frame in frames], rotation=45)
    plt.xlabel("Frame index")
    plt.ylabel("MPJPE (px)")
    plt.title("Mean Per Joint Position Error by Frame and Camera (Full Scale)")
    plt.legend()

    ax = plt.gca()
    apply_y_ticks(ax, mpjpes, zoom=False, is_mse=False)

    plt.savefig(os.path.join(output_dir, "mpjpe_plot.png"), dpi=200)
    plt.close()

    # ========== MPJPE PLOT (ZOOMED) ==========
    threshold = np.percentile(mpjpes, 95)
    plt.figure(figsize=(12, 5))
    for cam in sorted(set(cameras)):
        cam_frames = [f for f, c in zip(frames, cameras) if c == cam]
        cam_vals = [m for m, c in zip(mpjpes, cameras) if c == cam]
        plt.plot(cam_frames, cam_vals, 'o-', color=colors.get(cam, 'black'), label=f'Camera {cam}')

    plt.xticks(frames, [f"{frame}" for frame in frames], rotation=45)
    plt.ylim(0, threshold)
    plt.xlabel("Frame index")
    plt.ylabel("MPJPE (px)")
    plt.title(f"Mean Per Joint Position Error (Zoomed, <= {threshold:.1f}px)")
    plt.legend()

    ax = plt.gca()
    apply_y_ticks(ax, mpjpes, zoom=True, is_mse=False)

    plt.savefig(os.path.join(output_dir, "mpjpe_plot_zoom.png"), dpi=200)
    plt.close()

    # ========== MSE PLOT (FULL) ==========
    plt.figure(figsize=(12, 5))
    for cam in sorted(set(cameras)):
        cam_frames = [f for f, c in zip(frames, cameras) if c == cam]
        cam_vals = [m for m, c in zip(mses, cameras) if c == cam]
        plt.plot(cam_frames, cam_vals, 'o-', color=colors.get(cam, 'black'), label=f'Camera {cam}')

    plt.xticks(frames, [f"{frame}" for frame in frames], rotation=45)
    plt.xlabel("Frame index")
    plt.ylabel("MSE (px²)")
    plt.title("Mean Squared Error by Frame and Camera (Full Scale)")
    plt.legend()

    ax = plt.gca()
    apply_y_ticks(ax, mses, zoom=False, is_mse=True)

    plt.savefig(os.path.join(output_dir, "mse_plot.png"), dpi=200)
    plt.close()

    # ========== MSE PLOT (ZOOMED) ==========
    threshold = np.percentile(mses, 95)
    plt.figure(figsize=(12, 5))
    for cam in sorted(set(cameras)):
        cam_frames = [f for f, c in zip(frames, cameras) if c == cam]
        cam_vals = [m for m, c in zip(mses, cameras) if c == cam]
        plt.plot(cam_frames, cam_vals, 'o-', color=colors.get(cam, 'black'), label=f'Camera {cam}')

    plt.xticks(frames, [f"{frame}" for frame in frames], rotation=45)
    plt.ylim(0, threshold)
    plt.xlabel("Frame index")
    plt.ylabel("MSE (px²)")
    plt.title(f"Mean Squared Error (Zoomed, <= {threshold:.1f})")
    plt.legend()

    ax = plt.gca()
    apply_y_ticks(ax, mses, zoom=True, is_mse=True)

    plt.savefig(os.path.join(output_dir, "mse_plot_zoom.png"), dpi=200)
    plt.close()

# ==== MAIN ====
def main():
    # Load camera calibration
    cameras = {}
    for cam_idx in CAMERA_INDEXES:
        calib_path = os.path.join(CAMERA_MATRICES_DIR, f"cam_{cam_idx}", "calib", "camera_calib.json")
        mtx, dist, tvecs, rvecs = load_calibration(calib_path)
        cameras[cam_idx] = {"mtx": mtx, "dist": dist, "tvecs": tvecs, "rvecs": rvecs}

    # Map frames to GT label paths
    frame_to_labels = defaultdict(dict)
    for label_path in glob.glob(os.path.join(LABELS_DIR, "*.txt")):
        match = re.search(r'out(\d+)_frame_(\d+)', os.path.basename(label_path))
        if match:
            cam_idx = int(match.group(1))
            frame_idx = int(match.group(2))
            frame_to_labels[frame_idx][cam_idx] = label_path

    all_errors = []

    for tri_file in sorted(glob.glob(os.path.join(TRIANG_DIR, "*.txt"))):
        frame_idx = int(re.search(r'(\d+)', os.path.basename(tri_file)).group(1))
        points_3d = np.loadtxt(tri_file, dtype=np.float32)

        for cam_idx in CAMERA_INDEXES:
            if cam_idx not in frame_to_labels[frame_idx]:
                continue

            gt_keypoints = parse_annotation_file(frame_to_labels[frame_idx][cam_idx], IMG_WIDTH, IMG_HEIGHT)
            mtx, dist = cameras[cam_idx]["mtx"], cameras[cam_idx]["dist"]
            rvecs, tvecs = cameras[cam_idx]["rvecs"], cameras[cam_idx]["tvecs"]

            # Reproject
            reprojected_2d = reproject_points(points_3d, mtx, dist, rvecs, tvecs)
            reproj_keypoints = []
            errors = []
            for (gt_x, gt_y, v), (pr_x, pr_y) in zip(gt_keypoints, reprojected_2d):
                if v > 0:
                    errors.append(np.linalg.norm([gt_x - pr_x, gt_y - pr_y]))
                    reproj_keypoints.append((pr_x, pr_y, 1))
                else:
                    reproj_keypoints.append((pr_x, pr_y, 0))

            if errors:
                mpjpe = np.mean(errors)
                mse = np.mean(np.square(errors))
                all_errors.append({"frame": frame_idx,  "camera": cam_idx,"mpjpe": mpjpe,"mse": mse })
                print(f"Frame {frame_idx}, Cam {cam_idx}: MPJPE={mpjpe:.2f}px, MSE={mse:.2f}")

            # Draw overlay
            pattern = os.path.join(IMAGES_DIR, f"out{cam_idx}_frame_{frame_idx:04d}*.jpg")
            matches = glob.glob(pattern)
            if matches:
                image_path = matches[0]
            else:
                print(f"No rectified image found for cam {cam_idx}, frame {frame_idx}")
                continue

            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                overlay_img = draw_skeleton_overlay(img, gt_keypoints, reproj_keypoints)
                out_path = os.path.join(OUTPUT_OVERLAYS_DIR, f"overlay_frame_{frame_idx}_cam_{cam_idx}.jpg")
                cv2.imwrite(out_path, overlay_img)

    if all_errors:
        all_mpjpe = np.mean([e["mpjpe"] for e in all_errors])
        all_mse = np.mean([e["mse"] for e in all_errors])
        print(f"\nOverall MPJPE: {all_mpjpe:.2f}px")
        print(f"Overall MSE: {all_mse:.2f}px^2")
        plot_errors(all_errors, OUTPUT_PLOTS_DIR)

if __name__ == "__main__":
    main()