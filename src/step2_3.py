import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import glob
import os
import utils.config as config
import utils.calibration_utils as calib
import utils.annotation_utils as annot


# Parameters
VISUAL = False #if TRUE shows ALL 3D PLOTS and the ANIMATION, if FALSE just saves the 3d plots and the animation

# Load points
def plot_skeleton(frame_file, output_folder, visual=VISUAL):
    if not os.path.exists(frame_file):
        raise FileNotFoundError(f"File not found: {frame_file}")

    # Load points
    points_3d = np.loadtxt(frame_file)  # shape (num_joints, 3)

    # Prepare figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot keypoints
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
               c='red', s=50, label='Keypoints')

    # Plot bones
    for a, b in config.CONNECTIONS:
        if a < len(points_3d) and b < len(points_3d):
            p1, p2 = points_3d[a], points_3d[b]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'b-', linewidth=2)

    # Label joints
    for i, point in enumerate(points_3d):
        ax.text(point[0], point[1], point[2], str(i), fontsize=8)

    # Axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    frame_number = os.path.splitext(os.path.basename(frame_file))[0].split("_")[-1]
    ax.set_title(f"Skeleton Frame {frame_number}")
    ax.legend()

    # Equal aspect ratio
    max_range = np.max(np.ptp(points_3d, axis=0)) / 2
    mid_x = (np.max(points_3d[:, 0]) + np.min(points_3d[:, 0])) / 2
    mid_y = (np.max(points_3d[:, 1]) + np.min(points_3d[:, 1])) / 2
    mid_z = (np.max(points_3d[:, 2]) + np.min(points_3d[:, 2])) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    if VISUAL:
        # Show interactive plot
        plt.show()
    else:
        # Save output
        os.makedirs(output_folder, exist_ok=True)
        out_file = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(frame_file))[0]}.png")
        plt.savefig(out_file, dpi=200)
        plt.close(fig)
        print(f" Saved {out_file}")


# Batch process all frames
def process_all_frames(input_folder=config.TRIANG_FOLDER, output_folder=config.TRIANG_3D_FOLDER):
    frame_files = sorted(glob.glob(os.path.join(input_folder, "triangulated_frame_*.txt")))
    
    for frame_file in frame_files:
        plot_skeleton(frame_file, output_folder, visual=VISUAL)

def animate_skeleton(frame_folder=config.TRIANG_FOLDER,
                     n_frames=48,
                     visual=VISUAL,
                     output_file="../results/triang_3d_animation.mp4",
                     fps=12,
                     scale=1.0):

    # Load all frames
    all_frames = []
    for i in range(1, n_frames+1):
        fname = os.path.join(frame_folder, f"triangulated_frame_{i:04d}.txt")
        points = np.loadtxt(fname)
        points *= scale   # <--- apply scale
        all_frames.append(points)
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Fix axis limits for all frames
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
    lines = [ax.plot([], [], [], 'b-', linewidth=2)[0] for _ in config.CONNECTIONS]
    
    def init():
        scat._offsets3d = ([], [], [])
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return [scat] + lines
    
    def update(frame_idx):
        points = all_frames[frame_idx]
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        scat._offsets3d = (x, y, z)
        for k, (a, b) in enumerate(config.CONNECTIONS):
            lines[k].set_data([points[a, 0], points[b, 0]],
                              [points[a, 1], points[b, 1]])
            lines[k].set_3d_properties([points[a, 2], points[b, 2]])
        return [scat] + lines
    
    # Loop animation forever in interactive mode
    anim = FuncAnimation(fig,
                         update,
                         frames=n_frames,
                         init_func=init,
                         blit=False,
                         interval=1000/fps,
                         repeat=True)
    
    if not visual:
        anim.save(output_file, fps=fps, dpi=200)
        print(f" Animation saved to {output_file}")
    else:
        plt.show()

process_all_frames()
animate_skeleton(n_frames=48, visual=VISUAL, output_file="../results/triang_3d_animation.mp4", fps=12, scale=1.0)