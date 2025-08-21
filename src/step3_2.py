import cv2
import numpy as np
import scipy.io as sio
import os
import json
import utils.config as config
import utils.calibration_utils as calib
import utils.annotation_utils as annot

CAMERA_INDEXES = [2]

# Parameters
# ==========================
mocap_file = "../resources/mocap/Nick_2.mat"
video_file = "../results/rectified_videos/out2.mp4"
output_file = "../results/video_mocap_time_aligned.mp4"

# Alignment Parameters
# ==========================
# Align video frame 46 (0-indexed) with mocap frame 10475 (0-indexed)
VIDEO_ALIGN_FRAME = 45 # Frame 46 (0-indexed is 45)
MOCAP_ALIGN_FRAME = 9964 # MoCap frame 9965 (0-indexed is 9964) or 10475 (0-indexed is 10474)

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

# Load camera calibration
cameras = {}
for cam_idx in CAMERA_INDEXES:
    calib_path = os.path.join(config.RECT_CAMERA_FOLDER, f"cam_{cam_idx}_calib.json")
    mtx, dist, tvecs, rvecs, K_rect = load_calibration(calib_path)
    cameras[cam_idx] = {"mtx": K_rect, "dist": dist, "tvecs": tvecs, "rvecs": rvecs}


# Helper: Reproject 3D -> 2D
# ==========================
def reproject_points(points_3d, mtx, dist, rvecs, tvecs):
    points_2d, _ = cv2.projectPoints(points_3d, rvecs, tvecs, mtx, dist)
    return points_2d.reshape(-1, 2)

# Load MoCap data
# ==========================
data = sio.loadmat(mocap_file, squeeze_me=True, struct_as_record=False)
nick = data['Nick_2']

position_data = nick.Skeletons.PositionData  # shape [3, 24, 12000]
segment_labels = [str(lbl) for lbl in nick.Skeletons.SegmentLabels]

# MoCap metadata
mocap_frame_rate = 100
mocap_frames = 12000

# 
# Load Video
# ==========================
cap = cv2.VideoCapture(video_file)
rgb_frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
n_frames_rgb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Scale factor to map RGB frames to MoCap frames
# NEW: Calculate scale based on frame rates, not total frames
frame_scale = mocap_frame_rate / rgb_frame_rate

# Video writer
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, rgb_frame_rate, (width, height))

# Define Skeleton Connections
# ==========================
skeleton_edges = [
    ("Hips", "Spine"), ("Spine", "Spine1"), ("Spine1", "Spine2"), ("Spine2", "Neck"),
    ("Neck", "Head"),
    ("LeftShoulder", "LeftArm"), ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
    ("RightShoulder", "RightArm"), ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand"),
    ("Hips", "LeftUpLeg"), ("LeftUpLeg", "LeftLeg"), ("LeftLeg", "LeftFoot"),
    ("Hips", "RightUpLeg"), ("RightUpLeg", "RightLeg"), ("RightLeg", "RightFoot")
]

label_to_index = {lbl: i for i, lbl in enumerate(segment_labels)}

# ==========================
# Process Video
# ==========================
frame_idx = 0

calib_path = os.path.join(config.RECT_CAMERA_FOLDER, f"cam_{cam_idx}_calib.json")
mtx, dist, tvecs, rvecs , K_rect = load_calibration(calib_path)

mocap_idx_start = 0
mocap_idx_end = 999
    
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Find corresponding MoCap frame
    # NEW: Adjust calculation for alignment
    relative_video_frame = frame_idx - VIDEO_ALIGN_FRAME
    mocap_idx = int(MOCAP_ALIGN_FRAME + (relative_video_frame * frame_scale))
    mocap_idx = max(0, min(mocap_idx, mocap_frames - 1)) # Ensure index is within bounds

    if frame_idx == 0:
        mocap_idx_start=mocap_idx
    if frame_idx == 99:
        mocap_idx_end = mocap_idx

    # Extract skeleton joints (3D → shape [24, 3])
    joints_3d = position_data[:, :, mocap_idx].T  # (24, 3)

    # Reproject to 2D
    joints_2d = reproject_points(joints_3d, K_rect, dist, rvecs, tvecs)

    # Draw joints
    for (x, y) in joints_2d:
        cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

    # Draw bones
    for a, b in skeleton_edges:
        if a in label_to_index and b in label_to_index:
            p1 = joints_2d[label_to_index[a]]
            p2 = joints_2d[label_to_index[b]]
            cv2.line(frame, (int(p1[0]), int(p1[1])),
                                 (int(p2[0]), int(p2[1])),
                                 (255, 0, 0), 2)

    # Write frame
    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(" Skeleton overlay video saved:", output_file)
print(f"MOCAP START:",{ mocap_idx_start})
print(f"MOCAP END:",{ mocap_idx_end})
