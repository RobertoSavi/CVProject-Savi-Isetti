import numpy as np
import scipy.io as sio
import cv2
import os
from vedo import Plotter, Sphere, Line, Text2D

# ==========================
# Skeleton connections (bones)
# ==========================
skeleton_edges = [
    (0,1), (1,2), (2,3), (3,4), (4,5),
    (4,6), (6,7), (7,8), (8,10),
    (4,11), (11,12), (12,13), (13,15),
    (0,16), (16,17), (17,18), (18,19),
    (0,20), (20,21), (21,22), (22,23)
]

# ==========================
# Parameters
# ==========================
mocap_file = "Motion_Capture_Data/Nick_2.mat"
speed_factor = 1.0
video_filename = "skeleton_animation.mp4"
frame_rate = 25
tmp_folder = "tmp_frames"

# ==========================
# Load MoCap data
# ==========================
data = sio.loadmat(mocap_file, squeeze_me=True, struct_as_record=False)
nick = data['Nick_2']
positions = nick.Skeletons.PositionData
n_frames = positions.shape[2]

# ==========================
# Vedo Plotter
# ==========================
vp = Plotter(title="MoCap Skeleton", axes=1, size=(1000, 800))
joints = positions[:, :, 0].T
joint_spheres = [Sphere(pos, r=30, c="red") for pos in joints]
bone_lines = [Line(joints[i], joints[j], c="blue", lw=3) for i, j in skeleton_edges]
frame_text = Text2D(f"Frame: 0/{n_frames}", pos="top-right", s=1.5, c="black")
actors = joint_spheres + bone_lines + [frame_text]
vp.show(*actors, resetcam=True, interactive=False)


# Camera setup: above the head, 45° front-right
cam = vp.camera

distance = 5000       # distance from skeleton along y-axis (front)
angle_offset = -2000    # x-offset for 45° rotation to the right
height = 3500         # z-coordinate above skeleton head

# Set camera position
cam.SetPosition(angle_offset, distance, height)  

# Look at skeleton center (roughly middle of spine)
cam.SetFocalPoint(0, 0, 500)                    

# Keep vertical up
cam.SetViewUp(0, 0, 1)

vp.render()


# ==========================
# Prepare folder for temporary frames
# ==========================
if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)

# ==========================
# Render and save frames
# ==========================
for f in range(n_frames):
    joints = positions[:, :, f].T
    for i, s in enumerate(joint_spheres):
        s.pos(joints[i])

    for line in bone_lines:
        vp.remove(line)
    bone_lines = [Line(joints[i], joints[j], c="blue", lw=3) for (i, j) in skeleton_edges]

    current_frame = int(f * speed_factor)
    frame_text.text(f"Frame: {current_frame}/{n_frames}")

    vp.show(*joint_spheres, *bone_lines, frame_text, resetcam=False, interactive=False)

    # save frame as PNG
    filename = os.path.join(tmp_folder, f"frame_{f:05d}.png")
    vp.screenshot(filename)

# ==========================
# Combine frames into video using OpenCV
# ==========================
img_example = cv2.imread(os.path.join(tmp_folder, "frame_00000.png"))
height, width, _ = img_example.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_filename, fourcc, frame_rate, (width, height))

for f in range(n_frames):
    img_path = os.path.join(tmp_folder, f"frame_{f:05d}.png")
    img = cv2.imread(img_path)
    out.write(img)

out.release()
print(f"Video saved as {video_filename}")
