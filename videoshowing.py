import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import pandas as pd
from ultralytics import YOLO

MOCAP_FRAMES = 12000
MOCAP_FRAME_RATE = 100
RGB_FRAME_RATE = 25

fn = []

def get_trajectory_yolo():

    # Load the YOLOv8-pose model
    model = YOLO("utils/yolov8n-pose.pt")  # or yolov8s-pose.pt


    # Load your video
    video_path = "resources/out2.mp4"
    cap = cv2.VideoCapture(video_path)

    # Store keypoints over time
    frame_data = []

    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking", 960, 540)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_width = frame.shape[1]
        midpoint = frame_width // 2

        # Run pose estimation
        results = model(frame, conf=0.3)[0]

        # Get all detections
        people = results.keypoints.xy  # list of tensors [num_people, num_kpts, 2]

        rightmost_person = None
        max_x = -1

        for p in people:
            # Compute person's center x position based on average x of shoulder keypoints (5 and 6)
            kp = p.cpu().numpy()  # shape (17, 2)
            if kp.shape[0] < 11:  # skip incomplete detections
                continue

            person_x_center = (kp[5][0] + kp[6][0]) / 2  # avg of left/right shoulder

            # Only consider person whose center is on the right half
            if person_x_center > midpoint and person_x_center > max_x:
                rightmost_person = kp
                max_x = person_x_center

        # If a person on the right was found, extract keypoints
        if rightmost_person is not None:
            right_shoulder = rightmost_person[6]
            right_elbow = rightmost_person[8]
            right_wrist = rightmost_person[10]

            # Save the keypoints
            frame_data.append({
                "frame": frame_idx,
                "right_shoulder_x": right_shoulder[0],
                "right_shoulder_y": right_shoulder[1],
                "right_elbow_x": right_elbow[0],
                "right_elbow_y": right_elbow[1],
                "right_wrist_x": right_wrist[0],
                "right_wrist_y": right_wrist[1],
            })

            # Optional: draw skeleton
            for i, point in enumerate([right_shoulder, right_elbow, right_wrist]):
                x, y = int(point[0]), int(point[1])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            cv2.line(frame, tuple(right_shoulder.astype(int)), tuple(right_elbow.astype(int)), (255, 0, 0), 2)
            cv2.line(frame, tuple(right_elbow.astype(int)), tuple(right_wrist.astype(int)), (255, 0, 0), 2)

        # Display
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # Save data to CSV
    df = pd.DataFrame(frame_data)
    df.to_csv("right_arm_keypoints.csv", index=False)
    print("Saved right arm keypoints to CSV.")

def plot_keypoint_data():
    global fn

    # Load keypoint data
    df = pd.read_csv("right_arm_keypoints.csv")

    # Extract relevant columns
    frames = df["frame"]
    wrist_y = df["right_wrist_y"]

    min_y_index = wrist_y.idxmin()
    frame_number = df.loc[min_y_index, "frame"]
    fn = frame_number
    wrist_y_value = df.loc[min_y_index, "right_wrist_y"]
    timestamp_seconds = frame_number / RGB_FRAME_RATE
    time_seconds = frames / RGB_FRAME_RATE
    # --- Plot vs Frame ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(frames, wrist_y, marker='o', linewidth=2)
    plt.title("Right Wrist Y vs Frame")
    plt.xlabel("Frame Number")
    plt.ylabel("Wrist Y Position (pixels)")
    plt.gca().invert_yaxis()
    plt.grid(True)

    # --- Plot vs Time ---
    plt.subplot(1, 2, 2)
    plt.plot(time_seconds, wrist_y, marker='o', linewidth=2)
    plt.title("Right Wrist Y vs Time (seconds)")
    plt.xlabel("Time (s)")
    plt.ylabel("Wrist Y Position (pixels)")
    plt.gca().invert_yaxis()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"Max wrist height: {wrist_y_value:.2f} pixels")
    print(f"At frame: {frame_number}, which is {timestamp_seconds:.2f} seconds")


def save_frame_from_video(frame_number):
    # Open the video
    video_path = "resources/out2.mp4"
    output_path = "MaxWristHeight_Frame.png"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    success, frame = cap.read()

    if success:
        # Save the frame as an image
        cv2.imwrite(output_path, frame)
        print(f"Frame {frame_number} saved as {output_path}")
    else:
        print(f"Error: Could not read frame {frame_number}.")

    cap.release()


get_trajectory_yolo()
plot_keypoint_data()
save_frame_from_video(fn)







