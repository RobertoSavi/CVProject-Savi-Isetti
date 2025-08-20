import cv2
import numpy as np
import scipy.io as sio
import os
import json
import step2_4
import step2_1

CAMERA_INDEXES = [2]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================
# Parameters
# ==========================
mocap_file = "Motion_Capture_Data/Nick_2.mat"
video_file = "Motion_Capture_Data/full_video.mp4"
output_file = "output_skeleton_overlay2.mp4"

# Load camera calibration
cameras = {}
for cam_idx in CAMERA_INDEXES:
    calib_path = os.path.join(step2_1.RECT_CAMERA_FOLDER, f"cam_{cam_idx}_calib.json")
    mtx, dist, tvecs, rvecs, R_rect = step2_1.load_calibration(calib_path)
    cameras[cam_idx] = {"mtx": R_rect, "dist": dist, "tvecs": tvecs, "rvecs": rvecs}


# ==========================
# Helper: Reproject 3D -> 2D
# ==========================
def reproject_points(points_3d, mtx, dist, rvecs, tvecs):
    points_2d, _ = cv2.projectPoints(points_3d, rvecs, tvecs, mtx, dist)
    return points_2d.reshape(-1, 2)

# ==========================
# Load MoCap data
# ==========================
data = sio.loadmat(mocap_file, squeeze_me=True, struct_as_record=False)
nick = data['Nick_2']

position_data = nick.Skeletons.PositionData  # shape [3, 24, 12000]
segment_labels = [str(lbl) for lbl in nick.Skeletons.SegmentLabels]

# MoCap metadata
mocap_frame_rate = int(nick.FrameRate)
mocap_frames = position_data.shape[2]

# ==========================
# Load Video
# ==========================
cap = cv2.VideoCapture(video_file)
rgb_frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
n_frames_rgb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Scale factor to map RGB frames to MoCap frames
frame_scale = mocap_frames / n_frames_rgb

# Video writer
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, rgb_frame_rate, (width, height))

# ==========================
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

calib_path = os.path.join(step2_1.RECT_CAMERA_FOLDER, f"cam_{cam_idx}_calib.json")
mtx, dist, tvecs, rvecs, _ = step2_1.load_calibration(calib_path)
    
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Find corresponding MoCap frame
    mocap_idx = int(frame_idx * frame_scale)
    mocap_idx = min(mocap_idx, mocap_frames - 1)

    # Extract skeleton joints (3D → shape [24, 3])
    joints_3d = position_data[:, :, mocap_idx].T  # (24, 3)

    # Reproject to 2D
    #print(f"mtx:",{cameras[index]["mtx"]},"dist: ",{ cameras[index]["dist"]}, "rvecs: ",{cameras[index]["rvecs"]}, "tvecs: ",{cameras[index]["tvecs"]})
    joints_2d = reproject_points(joints_3d, mtx, dist, rvecs, tvecs)

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