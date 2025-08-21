import numpy as np
import scipy.io as sio
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from vedo import Plotter, Sphere, Line, Text2D
from step3_2 import mocap_idx_start,mocap_idx_end

VISUAL = False #if TRUE shows ALL 3D PLOTS and the ANIMATION, if FALSE just saves the 3d plots and the animation
CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4), (4,5),
    (6,7), (7,8), (8,9), (9,10),
    (11,12), (12,13), (13,14), (14,15),
    (0,16), (16,17), (17,18), (18,19),
    (0,20), (20,21), (21,22), (22,23)
]

def animate_mocap(mocap_file="../resources/mocap/Nick_2.mat",
                  start_frame=0,
                  end_frame=500,
                  visual=True,                 # True = show, False = save
                  output_file="../results/mocap_3d_animation.mp4",
                  fps=100,
                  scale=1.0):
    
    # Load MoCap data
    data = sio.loadmat(mocap_file, squeeze_me=True, struct_as_record=False)
    nick = data['Nick_2']
    positions = nick.Skeletons.PositionData   # shape [3, 24, Nframes]
    n_frames_total = positions.shape[2]
    
    # Clip frame range
    start_frame = max(0, start_frame)
    end_frame = min(n_frames_total-1, end_frame)
    frames = range(start_frame, end_frame+1)

    # Preload all frames
    all_frames = []
    for f in frames:
        pts = positions[:, :, f].T * scale  # (24, 3)
        all_frames.append(pts)
    
    # Matplotlib 3D setup
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Fix axis ranges for stability
    all_points = np.vstack(all_frames)
    max_range = np.max(np.ptp(all_points, axis=0)) / 2
    mid_x = (np.max(all_points[:, 0]) + np.min(all_points[:, 0])) / 2
    mid_y = (np.max(all_points[:, 1]) + np.min(all_points[:, 1])) / 2
    mid_z = (np.max(all_points[:, 2]) + np.min(all_points[:, 2])) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    scat = ax.scatter([], [], [], c='red', s=50)
    lines = [ax.plot([], [], [], 'b-', linewidth=2)[0] for _ in CONNECTIONS]

    def init():
        scat._offsets3d = ([], [], [])
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return [scat] + lines

    def update(frame_idx):
        points = all_frames[frame_idx - start_frame]  # local indexing
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        scat._offsets3d = (x, y, z)
        for k, (a, b) in enumerate(CONNECTIONS):
            lines[k].set_data([points[a, 0], points[b, 0]],
                              [points[a, 1], points[b, 1]])
            lines[k].set_3d_properties([points[a, 2], points[b, 2]])
        ax.set_title(f"MoCap Frame: {frame_idx}/{n_frames_total}")
        return [scat] + lines

    anim = FuncAnimation(fig,
                         update,
                         frames=frames,
                         init_func=init,
                         blit=False,
                         interval=1000/fps,
                         repeat=False)

    if visual:
        plt.show()
    else:
        anim.save(output_file, fps=fps, dpi=200)
        print(f" Animation saved to {output_file}")

animate_mocap(start_frame=mocap_idx_start, end_frame=mocap_idx_end, visual=VISUAL, fps=100)
