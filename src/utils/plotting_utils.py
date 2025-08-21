import numpy as np
import cv2

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