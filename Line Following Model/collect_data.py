import os
import cv2
import numpy as np

def main():
    images_dir = "extracted_images"
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    # Parameters
    ROI_BOTTOM_FRACTION = 0.60
    FEATURE_HEIGHT = 15
    FEATURE_WIDTH = 30
    LOWER_GREEN = np.array([35, 30, 20])  # Optimized to handle dark shadows under bridge
    UPPER_GREEN = np.array([90, 255, 255])
    MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    MIN_CONTOUR_AREA = 150
    STEERING_GAIN = 1.5
    SAMPLE_STEP = 20  # Sample every 20th frame to get ~1800 diverse frames

    # Fallback to Desktop path if directory doesn't exist locally
    if not os.path.exists(images_dir):
        images_dir = "C:/Users/G7/Desktop/SoSe-2026-Machine-Learning---Team-8/extracted_images"

    if not os.path.exists(images_dir):
        print(f"Error: {images_dir} does not exist!")
        return

    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    print(f"Found {len(image_files)} images. Sampling every {SAMPLE_STEP}th frame...")

    features_list = []
    labels_list = []

    for idx, fname in enumerate(image_files):
        if idx % SAMPLE_STEP != 0:
            continue

        img_path = os.path.join(images_dir, fname)
        image = cv2.imread(img_path)
        if image is None:
            continue

        h, w = image.shape[:2]
        roi_start_row = int(h * (1.0 - ROI_BOTTOM_FRACTION))
        roi = image[roi_start_row:, :]
        roi_h, roi_w = roi.shape[:2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < MIN_CONTOUR_AREA:
            continue

        # Centroid method for robust target steering
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            continue
        line_x = M["m10"] / M["m00"]

        image_center_x = roi_w / 2.0
        offset = (line_x - image_center_x) / image_center_x
        target_steer = np.clip(offset * STEERING_GAIN, -1.0, 1.0)

        # Feature extraction: resize mask to 15x30 and binarize
        mask_resized = cv2.resize(mask, (FEATURE_WIDTH, FEATURE_HEIGHT), interpolation=cv2.INTER_AREA)
        features = (mask_resized > 0).astype(np.float32).flatten()

        features_list.append(features)
        labels_list.append(target_steer)

    features_arr = np.array(features_list, dtype=np.float32)
    labels_arr = np.array(labels_list, dtype=np.float32)

    np.save(os.path.join(output_dir, "features.npy"), features_arr)
    np.save(os.path.join(output_dir, "labels.npy"), labels_arr)

    print(f"Data collection complete!")
    print(f"Saved {len(features_arr)} samples.")
    print(f"Features shape: {features_arr.shape}")
    print(f"Labels shape: {labels_arr.shape}")

    # Display class distribution based on discretization threshold of 0.10
    lefts = np.sum(labels_arr < -0.10)
    rights = np.sum(labels_arr > 0.10)
    straights = len(labels_arr) - lefts - rights
    print(f"Class distribution (Left / Straight / Right): {lefts} / {straights} / {rights}")

if __name__ == "__main__":
    main()
