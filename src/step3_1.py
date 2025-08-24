import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import pandas as pd
import torch
import os
import utils.config as config

MOCAP_FRAMES = 12000
MOCAP_FRAME_RATE = 100

good_shots = [99] # timestamps of good shots using right hand, in seconds 
windowSize = 2 # window's size in seconds for the plots, choose the window size that you prefere

mat = scipy.io.loadmat(
        os.path.join(config.MOCAP_FOLDER, "Nick_2.mat"),
        struct_as_record=False,
        squeeze_me=True
    )
nick = mat['Nick_2']

frame_rate = nick.FrameRate
position_data = nick.Skeletons.PositionData  # (3, 24, n_frames)
segment_labels = [str(label) for label in nick.Skeletons.SegmentLabels]
n_frames = MOCAP_FRAMES

# Index of the right and left hand
left_hand_idx = segment_labels.index('LeftHand')
right_hand_idx = segment_labels.index('RightHand')

print(f"total MoCap frames: {n_frames}, MoCap Frame rate: {frame_rate}")

def plotHeight(position_data, segment_labels, frame_rate, good_shot_time_s, window_s):
    n_frames = position_data.shape[2]

    # Get hand indices
    left_hand_idx = segment_labels.index('LeftHand')
    right_hand_idx = segment_labels.index('RightHand')

    # Convert to frame index
    center_frame = int(good_shot_time_s * frame_rate)
    window = int(window_s * frame_rate)
    start = max(0, center_frame - window)
    end = min(n_frames, center_frame + window)

    # Time x-axis in actual frame numbers
    x = np.arange(start, end)

    # Z trajectories
    left_z = position_data[2, left_hand_idx, start:end]
    right_z = position_data[2, right_hand_idx, start:end]

    # Find peak (max) of right hand height in this window
    peak_local_index = np.argmax(right_z)               # Index within the window
    peak_frame = x[peak_local_index]         
    peak_value = right_z[peak_local_index]           # Absolute frame number
    peak_time = peak_frame / frame_rate                 # Time in seconds

    # Store result

    right_hand_peak = []
    right_hand_peak.append((peak_frame, peak_time))

    # Optional: print result
    print(f"Good shot @ {good_shot_time_s:.3f}s (Frame {center_frame}) → "
          f"Right hand peak at Frame {peak_frame} → Time {peak_time:.3f}s")


    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(x, left_z, label='Left Hand Height (Z)', linewidth=2)
    plt.plot(x, right_z, label='Right Hand Height (Z)', linewidth=2)

    # Plot the peak point
    plt.plot(peak_frame, peak_value, 'ro', label='Right Hand Peak')
    
    plt.axvline(center_frame, color='gray', linestyle='--', label='Good Shot Frame')
    plt.xlabel('Frame Number')
    plt.ylabel('Z Position (Height)')
    plt.title('Hand Height vs Frame around Good Shot')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

for gs in good_shots:
    plotHeight(position_data=nick.Skeletons.PositionData,
            segment_labels=[str(lbl) for lbl in nick.Skeletons.SegmentLabels],
            frame_rate=nick.FrameRate,
            good_shot_time_s=gs,  # replace with timestamp
            window_s=windowSize)
    