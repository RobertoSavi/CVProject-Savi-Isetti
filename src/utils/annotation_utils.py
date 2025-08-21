
# Parse a YOLO annotation line into bbox and keypoints in pixel coordinates
def parse_annotation_file(label_path, img_w, img_h):
    with open(label_path, 'r') as f:
        label = f.readline().strip()
    parts = label.split()
    class_id = int(parts[0]) # Class ID of the object
    bbox = list(map(float, parts[1:5])) # The four points that define the bounding box
    kp_data = list(map(float, parts[5:])) # Keypoints data in YOLO format

    keypoints = []
    for i in range(0, len(kp_data), 3):
        x = kp_data[i] * img_w
        y = kp_data[i + 1] * img_h
        v = kp_data[i + 2]
        keypoints.append((x, y, v))
    # Also return the class ID, and bounding box alongside the keypoints to reconstruct the full annotation after rectification
    return class_id, bbox, keypoints

# Normalize keypoints to [0, 1] range for saving back to YOLO format
def normalize_keypoints(keypoints, img_w, img_h):
    normed = []
    for x, y, v in keypoints:
        normed.extend([x / img_w, y / img_h, v])
    return normed