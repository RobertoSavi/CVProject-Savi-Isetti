import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Raw extrinsics (flattened) ---
rvecs_flat = [
    1.7490662336349487, -0.7360826134681702, 0.5704466700553894,
    1.5034636699075041, 1.46262783446548, -0.9208994204122853,
    0.30208336376850653, -2.441891967305594, 1.6346525642920327,
    0.016147218644618988, 2.546968460083008, -1.7685179710388184
]

tvecs_flat = [
    7922.06103515625, -260.5303649902344, 24049.44140625,
    -1463.5382592705864, -2375.976286475501, 23369.00220453303,
    -9476.460800730425, -1564.631608918155, 23182.06472451103,
    -938.4185180664062, -210.0382537841797, 19418.685546875
]

# --- Reshape into (4 cameras, 3x1 vectors) ---
rvecs = [np.array(rvecs_flat[i:i+3], dtype=float).reshape(3,1) for i in range(0, len(rvecs_flat), 3)]
tvecs = [np.array(tvecs_flat[i:i+3], dtype=float).reshape(3,1) for i in range(0, len(tvecs_flat), 3)]

# --- Compute camera centers in world coords ---
camera_centers = []
for rvec, tvec in zip(rvecs, tvecs):
    R, _ = cv2.Rodrigues(rvec)
    C = -R.T @ tvec
    camera_centers.append(C.flatten())

camera_centers = np.array(camera_centers)

# === Matplotlib 3D visualization ===
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot cameras
ax.scatter(camera_centers[:,0], camera_centers[:,1], camera_centers[:,2], c='red', s=50, label='Cameras')

# Plot world origin
ax.scatter(0, 0, 0, c='blue', s=100, marker='x', label='Origin')

# Add labels for each camera
for i, (x,y,z) in enumerate(camera_centers):
    ax.text(x, y, z, f"Cam {i+1}", color='black')

# Axis labels
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")
ax.legend()

# Equal aspect ratio
max_range = np.array([camera_centers[:,0].max()-camera_centers[:,0].min(), 
                      camera_centers[:,1].max()-camera_centers[:,1].min(), 
                      camera_centers[:,2].max()-camera_centers[:,2].min()]).max() / 2.0

mid_x = (camera_centers[:,0].max()+camera_centers[:,0].min()) * 0.5
mid_y = (camera_centers[:,1].max()+camera_centers[:,1].min()) * 0.5
mid_z = (camera_centers[:,2].max()+camera_centers[:,2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.show()
