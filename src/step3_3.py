import os
import re
import glob
import json
import numpy as np
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
import utils.config as config
import utils.calibration_utils as calib
import utils.annotation_utils as annot
import utils.plotting_utils as plot

def get_mocap_idx(frame_idx, align_annotation=22, align_mocap=9965,
                  mocap_fps=100, ann_fps=12, n_mocap_frames=12000):
    scale = mocap_fps / ann_fps
    relative = frame_idx - align_annotation
    mocap_idx = int(round(align_mocap + relative * scale))
    if n_mocap_frames is not None:
        mocap_idx = max(0, min(mocap_idx, n_mocap_frames - 1))
    return mocap_idx
        

def align_mocap_and_annotations(img_path, keypoints, joints_3d, camera, out_path):
    # joints_3d: (24, 3) from your mocap file
    mc = np.asarray(joints_3d, dtype=float)

    # keypoints: list/array of (x, y, score) -> (N, 3)
    kp = np.asarray(keypoints, dtype=float)

    # Use label indices in sorted order (0..17)
    lbl_idxs = sorted(config.LABELS_CONVERTER.keys())           # e.g. [0,1,...,17]
    mc_idxs  = [config.LABELS_CONVERTER[i] for i in lbl_idxs]   # map to mocap indices

    # Aligned arrays
    kp_xy   = kp[lbl_idxs, :2]               # (18, 2)  keep x,y only (drop score)
    mc_xyz  = mc[np.array(mc_idxs), :3]      # (18, 3)

    # Build dict: {label_idx: ((x,y), (X,Y,Z))}
    pairs_dict = {
        idx: (tuple(kp_xy[i]), tuple(mc_xyz[i]))
        for i, idx in enumerate(lbl_idxs)
    }
    
    mtx, dist, tvecs, rvecs, K_rect, P = camera

    pairs = sorted(pairs_dict.items())  # ensure consistent order
    img_pts = np.array([p[1][0] for p in pairs], dtype=np.float64)   # (N,2)
    obj_pts = np.array([p[1][1] for p in pairs], dtype=np.float64)   # (N,3)

    distCoeffs = np.zeros((5, 1), dtype=np.float64)

    # Solve PnP: pose of camera w.r.t. mocap coordinates
    # RANSAC to handle potential outliers, then refine with ITERATIVE on inliers
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj_pts,
        imagePoints=img_pts,
        cameraMatrix=K_rect,
        distCoeffs=distCoeffs,
        flags=cv2.SOLVEPNP_AP3P,        # good robust initializer
        iterationsCount=200,
        reprojectionError=3.0,          # px
        confidence=0.999
    )
    if not ok or inliers is None or len(inliers) < 6:
        # Fallback to a non-RANSAC approach if needed
        ok, rvec, tvec = cv2.solvePnP(
            objectPoints=obj_pts,
            imagePoints=img_pts,
            cameraMatrix=K_rect,
            distCoeffs=distCoeffs,
            flags=cv2.SOLVEPNP_EPNP
        )
        inliers = np.arange(obj_pts.shape[0]).reshape(-1, 1)

    # Refine on inliers with Levenberg-Marquardt
    ok, rvec, tvec = cv2.solvePnP(
        objectPoints=obj_pts[inliers[:,0]],
        imagePoints=img_pts[inliers[:,0]],
        cameraMatrix=K_rect,
        distCoeffs=distCoeffs,
        rvec=rvec, tvec=tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    # Project mocap 3D points onto the image
    proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K_rect, distCoeffs)
    proj_pts = proj_pts.reshape(-1, 2)

    # Compute reprojection error (RMS and per-point)
    errors = np.linalg.norm(proj_pts - img_pts, axis=1)
    rms = np.sqrt(np.mean(errors**2))
    #print(f"PnP inliers: {len(inliers)}/{len(obj_pts)} | RMS reprojection error: {rms:.3f} px")

    # Visualize overlay
    img = cv2.imread(img_path)
    assert img is not None, f"Could not read image: {img_path}"

    # Draw GT 2D keypoints (blue) and projected mocap (red), and lines between them
    img = plot.draw_joints_and_skeleton(img, proj_pts, config.CONNECTIONS, color=(255, 0, 0))
    img = plot.draw_joints_and_skeleton(img, img_pts, config.CONNECTIONS, color=(0, 255, 0))
    img = plot.draw_skeleton_connections(img, img_pts, proj_pts, color=(0,255,255))    # yellow links
    # Put RMS text
    cv2.putText(img, f"RMS reprojection: {rms:.2f}px | Inliers: {len(inliers)}/{len(obj_pts)}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, f"RMS reprojection: {rms:.2f}px | Inliers: {len(inliers)}/{len(obj_pts)}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    # Save the overlay image
    cv2.imwrite(out_path, img)
    print("Saved overlay to:", out_path)
    return rvec, tvec, K_rect
    
# From the projection matrix P = K [R|t], recover the camera pose:
# World (W) -> Camera (C): T_C_from_W (4x4)
def _T_C_from_W_from_P(P):
    K, R, t_h, *_ = cv2.decomposeProjectionMatrix(P)
    C_W = (t_h[:3] / t_h[3]).reshape(3)  # camera center in world coords
    t = -R @ C_W                         # t_{C<-W}
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = t
    return T

# From PnP extrinsics (mocap -> camera), build inverse:
# Camera (C) -> Mocap (M): T_M_from_C (4x4)
def _T_M_from_C_from_pnp(rvecs, tvecs):
    R_C_from_M, _ = cv2.Rodrigues(rvecs)   # R_{C<-M}
    t_C_from_M = tvecs.reshape(3)          # t_{C<-M}
    R_M_from_C = R_C_from_M.T
    t_M_from_C = -R_M_from_C @ t_C_from_M
    T = np.eye(4); T[:3,:3] = R_M_from_C; T[:3,3] = t_M_from_C
    return T

# Apply 4x4 transform T to Nx3 points X.
def _transform_points(X, T):
    Xh = np.c_[X, np.ones((X.shape[0],1))]
    return (Xh @ T.T)[:, :3]

def _set_axes_equal(ax):
    xlim = ax.get_xlim3d(); ylim = ax.get_ylim3d(); zlim = ax.get_zlim3d()
    xr = abs(xlim[1] - xlim[0]); yr = abs(ylim[1] - ylim[0]); zr = abs(zlim[1] - zlim[0])
    r = max(xr, yr, zr)
    xmid = 0.5 * (xlim[0] + xlim[1])
    ymid = 0.5 * (ylim[0] + ylim[1])
    zmid = 0.5 * (zlim[0] + zlim[1])
    ax.set_xlim3d([xmid - r/2, xmid + r/2])
    ax.set_ylim3d([ymid - r/2, ymid + r/2])
    ax.set_zlim3d([zmid - r/2, zmid + r/2])


# Save a 3D plot of mocap vs triangulated points (both in mocap frame)
def plot_3d_mocap_vs_triangulation(mc_xyz, tri_in_M, output_path):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X (mocap)")
    ax.set_ylabel("Y (mocap)")
    ax.set_zlabel("Z (mocap)")

    # Draw mocap (blue circles) and triangulated (orange triangles)
    h1 = plot.draw_joints_and_skeleton_3d(ax, mc_xyz,     config.CONNECTIONS, color='tab:blue',   marker='o', s=30, lw=1.5, label="Mocap GT")
    h2 = plot.draw_joints_and_skeleton_3d(ax, tri_in_M,   config.CONNECTIONS, color='tab:orange', marker='^', s=30, lw=1.5, label="Triangulated→M")

    # Yellow connectors between corresponding joints
    plot.draw_skeleton_connections_3d(ax, mc_xyz, tri_in_M, color='y', lw=2.0, alpha=0.9)

    _set_axes_equal(ax)
    ax.legend(handles=[h1, h2], loc="upper right")
    plt.tight_layout()

    # Ensure directory exists and save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def evaluate_triangulation_error(camera, rvecs, tvecs, tri_xyz_W, mc_xyz, units="m"):
    _, _, _, _, K_rect, P = camera
    T_C_from_W = _T_C_from_W_from_P(P)           # W -> C
    T_M_from_C = _T_M_from_C_from_pnp(rvecs, tvecs)  # C -> M
    T_M_from_W = T_M_from_C @ T_C_from_W         # W -> M

    tri_in_M = _transform_points(tri_xyz_W, T_M_from_W)  # (N,3)

    to_mm = 1.0 if str(units).lower() == "mm" else 1000.0
    err_vec = tri_in_M - mc_xyz                 # in your native units
    per_joint = np.linalg.norm(err_vec, axis=1) * to_mm
    mpjpe = float(np.mean(per_joint))
    mse   = float(np.mean(per_joint**2))
    rmse  = float(np.sqrt(mse))

    return {
        "MPJPE_mm": mpjpe,
        "MSE_mm":   mse,
        "RMSE_mm" : rmse,
        "per_joint_mm": per_joint,
        "T_M_from_W": T_M_from_W,
        "T_M_from_C": T_M_from_C,
        "T_C_from_W": T_C_from_W,
        "tri_in_M": tri_in_M,
    } 
    
def main():
    os.makedirs(config.MOCAP_OVERLAYS_FOLDER, exist_ok=True)
    os.makedirs(config.MOCAP_TRIANG_3D_FOLDER, exist_ok=True)
    
    mat = scipy.io.loadmat(
        os.path.join(config.MOCAP_FOLDER, "Nick_2.mat"),
        struct_as_record=False,
        squeeze_me=True
    )
    nick = mat['Nick_2']
    frame_rate = nick.FrameRate
    position_data = nick.Skeletons.PositionData  # (3, 24, n_frames)
    

    all_errors = []
    for label_path in glob.glob(os.path.join(config.RECT_LABEL_FOLDER, "*.txt")):
        basename = os.path.basename(label_path)
        name_wo_ext = os.path.splitext(basename)[0]
        img_path = os.path.join(config.RECT_IMG_FOLDER, f"{name_wo_ext}.jpg")
        match = re.search(r'out(\d+)_frame_(\d+).*\.txt$', basename)
        if not match:
            print("Could not extract camera index or frame index from filename:", label_path)
            continue
        cam_index = int(match.group(1))
        frame_index = int(match.group(2))
        
        mocap_overlay_path = os.path.join(config.MOCAP_OVERLAYS_FOLDER, f"{name_wo_ext}_overlay.jpg")
        mocap_index = get_mocap_idx(frame_index, config.ANNOTATION_ALIGN_FRAME, config.MOCAP_ALIGN_FRAME,
                                    mocap_fps=frame_rate, ann_fps=12, n_mocap_frames=position_data.shape[2])
        joints_3d = position_data[:, :, mocap_index].T  # (24, 3)

        _, _, keypoints = annot.parse_annotation_file(label_path, config.IMG_WIDTH, config.IMG_HEIGHT)
        
        calib_path = os.path.join(config.RECT_CAMERA_FOLDER, f"cam_{cam_index}_calib.json")
        camera = calib.load_calibration(calib_path)
        
        rvec, tvec, K_rect = align_mocap_and_annotations(img_path, keypoints, joints_3d, camera, mocap_overlay_path)
        
        tri_path = os.path.join(config.TRIANG_FOLDER, f"triangulated_frame_{frame_index:04d}.txt")
        tri = np.loadtxt(tri_path, dtype=float)
        tri = tri.reshape(-1, 3)
        mc = np.asarray(joints_3d, dtype=float)     
        
        lbl_idxs   = sorted(config.LABELS_CONVERTER.keys())        # [0,1,2,...,17]
        mocap_idxs = [config.LABELS_CONVERTER[i] for i in lbl_idxs]

        tri_ordered = tri[lbl_idxs, :3]                     # (18,3)
        mc_ordered  = mc[np.array(mocap_idxs), :3]  

        result = evaluate_triangulation_error(camera, rvec, tvec, tri_ordered, mc_ordered, units="mm")
        # print(f"Frame {frame_index} | MPJPE: {result['MPJPE_mm']:.2f} mm | RMSE: {result['RMSE_mm']:.2f} mm")
        all_errors.append({"frame": frame_index,  "camera": cam_index,"mpjpe": result["MPJPE_mm"],"mse": result["MSE_mm"] })

        mocap_3d_path = os.path.join(config.MOCAP_TRIANG_3D_FOLDER, f"{name_wo_ext}_3D.png")
        plot_3d_mocap_vs_triangulation(
            mc_xyz=mc_ordered,
            tri_in_M=result["tri_in_M"],
            output_path=mocap_3d_path,
        )
    plot.plot_errors(all_errors, config.MOCAP_ERROR_PLOTS_FOLDER, unit="mm")

if __name__ == "__main__":
    main()