import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Draws joints and skeleton on an image
# Parameters
# img: Input image
# keypoints: (N,2) or (N,3) as (x, y[, v]); if v present, only v>0 are drawn
# connections: List of (a, b) index pairs to connect
# color: BGR color for both joints and bones
def draw_joints_and_skeleton(img, keypoints, connections, color=(0, 255, 0)):
    out = img.copy()
    kp = np.asarray(keypoints, dtype=float)
    has_vis = (kp.ndim == 2 and kp.shape[1] == 3)

    # Bones first
    for a, b in connections:
        if a < 0 or b < 0 or a >= len(kp) or b >= len(kp):
            continue
        if has_vis and (kp[a, 2] <= 0 or kp[b, 2] <= 0):
            continue
        xa, ya = kp[a, 0], kp[a, 1]
        xb, yb = kp[b, 0], kp[b, 1]
        if not np.isfinite([xa, ya, xb, yb]).all():
            continue
        cv2.line(out, (int(round(xa)), int(round(ya))),
                      (int(round(xb)), int(round(yb))),
                      color, 1)

    # Joints on top
    for i in range(len(kp)):
        if has_vis and kp[i, 2] <= 0:
            continue
        x, y = kp[i, 0], kp[i, 1]
        if not np.isfinite([x, y]).all():
            continue
        cv2.circle(out, (int(round(x)), int(round(y))), 3, color, -1)

    return out


# Draws yellow lines between two skeletons (point i to point i)
# Parameters
# img: Input image
# keypoints_a: (N,2) or (N,3) as (x, y[, v]); if v present, v>0 required
# keypoints_b: (N,2) or (N,3) as (x, y[, v]); if v present, v>0 required
# color: BGR color for the lines (default yellow)
def draw_skeleton_connections(img, keypoints_a, keypoints_b, color=(0, 255, 255)):
    out = img.copy()
    A = np.asarray(keypoints_a, dtype=float)
    B = np.asarray(keypoints_b, dtype=float)
    N = min(len(A), len(B))
    has_vis_a = (A.ndim == 2 and A.shape[1] == 3)
    has_vis_b = (B.ndim == 2 and B.shape[1] == 3)

    for i in range(N):
        if has_vis_a and A[i, 2] <= 0:
            continue
        if has_vis_b and B[i, 2] <= 0:
            continue
        xa, ya = A[i, 0], A[i, 1]
        xb, yb = B[i, 0], B[i, 1]
        if not np.isfinite([xa, ya, xb, yb]).all():
            continue
        cv2.line(out, (int(round(xa)), int(round(ya))),
                      (int(round(xb)), int(round(yb))),
                      color, 1)
    return out


# Draws joints and skeleton in 3D on a Matplotlib axis
# Parameters
# ax: Matplotlib 3D axis
# points: (N,3) or (N,4) as (x, y, z[, v]); if v present, only v>0 are drawn
# connections: List of (a, b) index pairs to connect
# color/marker/s/lw/label: Styling options; returns the scatter handle (for legend)
def draw_joints_and_skeleton_3d(ax, points, connections, color='tab:blue',
                                marker='o', s=30, lw=1.5, label=None):
    P = np.asarray(points, dtype=float)
    has_vis = (P.ndim == 2 and P.shape[1] == 4)

    # Bones first
    for a, b in connections:
        if a < 0 or b < 0 or a >= len(P) or b >= len(P):
            continue
        if has_vis and (P[a, 3] <= 0 or P[b, 3] <= 0):
            continue
        xa, ya, za = P[a, 0], P[a, 1], P[a, 2]
        xb, yb, zb = P[b, 0], P[b, 1], P[b, 2]
        if not np.isfinite([xa, ya, za, xb, yb, zb]).all():
            continue
        ax.plot([xa, xb], [ya, yb], [za, zb], color=color, linewidth=lw)

    # Joints on top
    if has_vis:
        mask = np.isfinite(P[:, 0]) & np.isfinite(P[:, 1]) & np.isfinite(P[:, 2]) & (P[:, 3] > 0)
    else:
        mask = np.isfinite(P[:, 0]) & np.isfinite(P[:, 1]) & np.isfinite(P[:, 2])

    h = ax.scatter(P[mask, 0], P[mask, 1], P[mask, 2],
                   s=s, marker=marker, color=color, depthshade=False, label=label)
    return h


# Draws yellow lines between two 3D skeletons (point i to point i)
# Parameters
# ax: Matplotlib 3D axis
# points_a: (N,3) or (N,4) as (x, y, z[, v]); if v present, v>0 required
# points_b: (N,3) or (N,4) as (x, y, z[, v]); if v present, v>0 required
# color/lw/alpha: Styling options for the connectors (default yellow)
def draw_skeleton_connections_3d(ax, points_a, points_b, color='y', lw=2.0, alpha=0.9):
    A = np.asarray(points_a, dtype=float)
    B = np.asarray(points_b, dtype=float)
    N = min(len(A), len(B))
    has_vis_a = (A.ndim == 2 and A.shape[1] == 4)
    has_vis_b = (B.ndim == 2 and B.shape[1] == 4)

    for i in range(N):
        if has_vis_a and A[i, 3] <= 0:
            continue
        if has_vis_b and B[i, 3] <= 0:
            continue
        xa, ya, za = A[i, 0], A[i, 1], A[i, 2]
        xb, yb, zb = B[i, 0], B[i, 1], B[i, 2]
        if not np.isfinite([xa, ya, za, xb, yb, zb]).all():
            continue
        ax.plot([xa, xb], [ya, yb], [za, zb], color=color, linewidth=lw, alpha=alpha)
        
        
# Plots MPJPE and MSE timelines (per frame and per camera) and saves 4 PNGs:
#   - mpjpe_plot.png          (full scale)
#   - mpjpe_plot_zoom.png     (y-axis zoomed to 95th percentile)
#   - mse_plot.png            (full scale)
#   - mse_plot_zoom.png       (y-axis zoomed to 95th percentile)
#
# Parameters
#   errors_list: list of dicts, each with keys:
#       'frame'  -> int frame index
#       'camera' -> int camera id (e.g., 2, 5, 8, 13)
#       'mpjpe'  -> float Mean Per Joint Position Error (in `unit`, e.g., mm)
#       'mse'    -> float Mean Squared Error (in `unit_sq`, e.g., mm²)
#   output_dir: directory where the PNGs will be written (created if missing)
#   unit:       label for MPJPE y-axis (e.g., "mm")
#   unit_sq:    label for MSE y-axis (defaults to f"{unit}²" if None)        
def plot_errors(errors_list, output_dir, unit="mm", unit_sq=None):
    if not errors_list:
        return
    if unit_sq is None:
        unit_sq = f"{unit}²"

    frames  = [e['frame']  for e in errors_list]
    cameras = [e['camera'] for e in errors_list]
    mpjpes  = [e['mpjpe']  for e in errors_list]
    mses    = [e['mse']    for e in errors_list]

    os.makedirs(output_dir, exist_ok=True)
    colors = {2: 'red', 5: 'blue', 8: 'green', 13: 'orange'}

    def apply_y_ticks(ax, data, zoom=False, is_mse=False):
        max_val = np.max(data) if len(data) else 0.0
        if zoom and len(data):
            max_val = np.percentile(data, 95)

        # heuristic step size depending on metric
        if is_mse:
            step = max(50, round(max_val / 15, -1))   # ~15 ticks, rounded to nearest 10
        else:
            step = max(5,  round(max_val / 15, -1))   # ~15 ticks, rounded to nearest 5

        ax.yaxis.set_major_locator(ticker.MultipleLocator(step))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax.grid(True, which="both", linestyle="--", alpha=0.6)

    # ========== MPJPE PLOT (FULL) ==========
    plt.figure(figsize=(12, 5))
    for cam in sorted(set(cameras)):
        cam_frames = [f for f, c in zip(frames, cameras) if c == cam]
        cam_vals   = [m for m, c in zip(mpjpes, cameras) if c == cam]
        plt.plot(cam_frames, cam_vals, 'o-', color=colors.get(cam, 'black'), label=f'Camera {cam}')

    plt.xticks(frames, [f"{frame}" for frame in frames], rotation=45)
    plt.xlabel("Frame index")
    plt.ylabel(f"MPJPE ({unit})")
    plt.title("Mean Per Joint Position Error by Frame and Camera (Full Scale)")
    plt.legend()

    ax = plt.gca()
    apply_y_ticks(ax, mpjpes, zoom=False, is_mse=False)

    plt.savefig(os.path.join(output_dir, "mpjpe_plot.png"), dpi=200)
    plt.close()

    # ========== MPJPE PLOT (ZOOMED) ==========
    threshold = np.percentile(mpjpes, 95) if len(mpjpes) else 0.0
    plt.figure(figsize=(12, 5))
    for cam in sorted(set(cameras)):
        cam_frames = [f for f, c in zip(frames, cameras) if c == cam]
        cam_vals   = [m for m, c in zip(mpjpes, cameras) if c == cam]
        plt.plot(cam_frames, cam_vals, 'o-', color=colors.get(cam, 'black'), label=f'Camera {cam}')

    plt.xticks(frames, [f"{frame}" for frame in frames], rotation=45)
    plt.ylim(0, threshold)
    plt.xlabel("Frame index")
    plt.ylabel(f"MPJPE ({unit})")
    plt.title(f"Mean Per Joint Position Error (Zoomed, <= {threshold:.1f} {unit})")
    plt.legend()

    ax = plt.gca()
    apply_y_ticks(ax, mpjpes, zoom=True, is_mse=False)

    plt.savefig(os.path.join(output_dir, "mpjpe_plot_zoom.png"), dpi=200)
    plt.close()

    # ========== MSE PLOT (FULL) ==========
    plt.figure(figsize=(12, 5))
    for cam in sorted(set(cameras)):
        cam_frames = [f for f, c in zip(frames, cameras) if c == cam]
        cam_vals   = [m for m, c in zip(mses, cameras) if c == cam]
        plt.plot(cam_frames, cam_vals, 'o-', color=colors.get(cam, 'black'), label=f'Camera {cam}')

    plt.xticks(frames, [f"{frame}" for frame in frames], rotation=45)
    plt.xlabel("Frame index")
    plt.ylabel(f"MSE ({unit_sq})")
    plt.title("Mean Squared Error by Frame and Camera (Full Scale)")
    plt.legend()

    ax = plt.gca()
    apply_y_ticks(ax, mses, zoom=False, is_mse=True)

    plt.savefig(os.path.join(output_dir, "mse_plot.png"), dpi=200)
    plt.close()

    # ========== MSE PLOT (ZOOMED) ==========
    threshold = np.percentile(mses, 95) if len(mses) else 0.0
    plt.figure(figsize=(12, 5))
    for cam in sorted(set(cameras)):
        cam_frames = [f for f, c in zip(frames, cameras) if c == cam]
        cam_vals   = [m for m, c in zip(mses, cameras) if c == cam]
        plt.plot(cam_frames, cam_vals, 'o-', color=colors.get(cam, 'black'), label=f'Camera {cam}')

    plt.xticks(frames, [f"{frame}" for frame in frames], rotation=45)
    plt.ylim(0, threshold)
    plt.xlabel("Frame index")
    plt.ylabel(f"MSE ({unit_sq})")
    plt.title(f"Mean Squared Error (Zoomed, <= {threshold:.1f} {unit_sq})")
    plt.legend()

    ax = plt.gca()
    apply_y_ticks(ax, mses, zoom=True, is_mse=True)

    plt.savefig(os.path.join(output_dir, "mse_plot_zoom.png"), dpi=200)
    plt.close()