import cv2

def save_video_slow(video_path, output_path, speed_factor=0.5):
    """
    Save a slowed-down version of the input video with frame number overlay.

    Parameters:
    - video_path (str): path to the input video file.
    - output_path (str): path to save the output video file.
    - speed_factor (float): playback speed factor (e.g., 0.2 = 20% speed).
    """

    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open input video.")
        return

    # Get video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define new fps for slowed down video
    new_fps = fps * speed_factor

    # Set up VideoWriter to save output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, new_fps, (width, height))

    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Overlay frame number (top-right)
        text = f"Frame: {frame_number}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 5
        thickness = 3
        color = (0, 255, 255)  # Yellow in BGR

        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = width - text_size[0] - 50
        y = 150
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

        # Write frame to output
        out.write(frame)
        frame_number += 1

    cap.release()
    out.release()
    print(f"Saved slowed video to: {output_path}")

save_video_slow("resources/out2.mp4", "resources/out2_slowed.mp4", speed_factor=0.2)
