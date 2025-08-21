import os
import re
import glob
import json
import numpy as np
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import utils.config as config
import utils.calibration_utils as calib
import utils.annotation_utils as annot
import utils.plotting_utils as plot

# ==== CONFIG ====

os.makedirs(config.TRIANG_OVERLAYS_FOLDER, exist_ok=True)
os.makedirs(config.TRIANG_ERROR_PLOTS_FOLDER, exist_ok=True)

def reproject_points(points_3d, mtx, dist, rvecs, tvecs):
    points_2d, _ = cv2.projectPoints(points_3d, rvecs, tvecs, mtx, dist)
    return points_2d.reshape(-1, 2)

def plot_errors(errors_list, output_dir):
    frames = [e['frame'] for e in errors_list]
    cameras = [e['camera'] for e in errors_list]
    mpjpes = [e['mpjpe'] for e in errors_list]
    mses = [e['mse'] for e in errors_list]

    colors = {2: 'red', 5: 'blue', 8: 'green', 13: 'orange'}

    def apply_y_ticks(ax, data, zoom=False, is_mse=False):
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
    for cam_idx in config.CAMERA_INDEXES:
        calib_path = os.path.join(config.RECT_CAMERA_FOLDER, f"cam_{cam_idx}_calib.json")
        mtx, dist, tvecs, rvecs, K_rect, P = calib.load_calibration(calib_path)

        if K_rect is not None:
            # Use rectified intrinsics, zero distortion
            K = np.asarray(K_rect, dtype=np.float32)
            dist_used = np.zeros((1, 5), dtype=np.float32)
        else:
            # Use original intrinsics + distortion
            K = np.asarray(mtx, dtype=np.float32)
            dist_used = np.asarray(dist, dtype=np.float32)

        cameras[cam_idx] = {
            "mtx": K,
            "dist": dist_used,
            "tvecs": np.asarray(tvecs, dtype=np.float32).reshape(3, 1),
            "rvecs": np.asarray(rvecs, dtype=np.float32).reshape(3, 1),
        }

    # Map frames to GT label paths
    frame_to_labels = defaultdict(dict)
    for label_path in glob.glob(os.path.join(config.RECT_LABEL_FOLDER, "*.txt")):
        match = re.search(r'out(\d+)_frame_(\d+)', os.path.basename(label_path))
        if match:
            cam_idx = int(match.group(1))
            frame_idx = int(match.group(2))
            frame_to_labels[frame_idx][cam_idx] = label_path

    all_errors = []

    for tri_file in sorted(glob.glob(os.path.join(config.TRIANG_FOLDER, "*.txt"))):
        frame_idx = int(re.search(r'(\d+)', os.path.basename(tri_file)).group(1))
        points_3d = np.loadtxt(tri_file, dtype=np.float32)

        for cam_idx in config.CAMERA_INDEXES:
            if cam_idx not in frame_to_labels[frame_idx]:
                continue

            _, _, gt_keypoints = annot.parse_annotation_file(frame_to_labels[frame_idx][cam_idx], config.IMG_WIDTH, config.IMG_HEIGHT)
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
            pattern = os.path.join(config.RECT_IMG_FOLDER, f"out{cam_idx}_frame_{frame_idx:04d}*.jpg")
            matches = glob.glob(pattern)
            if matches:
                image_path = matches[0]
            else:
                print(f"No rectified image found for cam {cam_idx}, frame {frame_idx}")
                continue

            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                img = plot.draw_joints_and_skeleton(img, gt_keypoints, config.CONNECTIONS, color=(0, 255, 0))
                img = plot.draw_joints_and_skeleton(img, reproj_keypoints, config.CONNECTIONS, color=(255, 0, 0))
                out_path = os.path.join(config.TRIANG_OVERLAYS_FOLDER, f"overlay_frame_{frame_idx}_cam_{cam_idx}.jpg")
                cv2.imwrite(out_path, img)

    if all_errors:
        all_mpjpe = np.mean([e["mpjpe"] for e in all_errors])
        all_mse = np.mean([e["mse"] for e in all_errors])
        print(f"\nOverall MPJPE: {all_mpjpe:.2f}px")
        print(f"Overall MSE: {all_mse:.2f}px^2")
        plot_errors(all_errors, config.TRIANG_ERROR_PLOTS_FOLDER)

if __name__ == "__main__":
    main()