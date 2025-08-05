import cv2
import os

def extract_clip_centered_on_frame(
    video_path,
    center_frame,
    window_seconds,
    output_path,
    crop_size=None  # e.g., (width, height)
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate start and end frame based on time window
    frames_before = int(fps * window_seconds)
    start_frame = max(0, center_frame - frames_before)
    end_frame = min(total_frames - 1, center_frame + frames_before)

    # Move to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Get frame dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Handle spatial cropping (optional)
    if crop_size:
        crop_width, crop_height = crop_size
        x0 = (width - crop_width) // 2
        y0 = (height - crop_height) // 2
    else:
        crop_width, crop_height = width, height
        x0, y0 = 0, 0

    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (crop_width, crop_height))

    # Read and write frames in the range
    for f in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break

        # Optional: spatial crop (center crop)
        cropped_frame = frame[y0:y0+crop_height, x0:x0+crop_width]

        out.write(cropped_frame)

    cap.release()
    out.release()
    print(f"Saved clip from frame {start_frame} to {end_frame} in: {output_path}")


extract_clip_centered_on_frame(
    video_path="resources/out2.mp4",
    center_frame=42,
    window_seconds=1,  # 1s before and after
    output_path="resources/out2_cropped.mp4",
    crop_size=None  # Optional: center-crop the frame (width, height)
)

