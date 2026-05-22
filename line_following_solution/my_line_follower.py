#!/usr/bin/env python3
"""
HSHL Line Following Student Lab — Your Implementation
=====================================================

Implement your line following algorithm by filling in the function below:

    detect_line(image)  — called for every camera frame (~30 fps)

─────────────────────────────────────────────────────────────────────────────
INPUTS  (what you receive from the camera)
─────────────────────────────────────────────────────────────────────────────
Camera frame  →  detect_line(image)
  image         np.ndarray, shape (720, 1280, 3), BGR colour order
                Same convention as OpenCV.

─────────────────────────────────────────────────────────────────────────────
OUTPUTS  (what your function must return)
─────────────────────────────────────────────────────────────────────────────
detect_line(image)  →  float | None
  Return a steering value in range [-1.0, 1.0]:
    -1.0  = steer full left
     0.0  = go straight (line is centered)
    +1.0  = steer full right
        None  = cannot detect line (framework uses neutral steering fallback)

─────────────────────────────────────────────────────────────────────────────
ALGORITHM TIPS
─────────────────────────────────────────────────────────────────────────────
This version uses an SVM Machine Learning model trained using OpenCV features.
It defaults to using the SVM model prediction probabilities for smooth steering.
If the model file is not found, it falls back to the robust contour moments method.
"""
import os
import time
import pickle
import cv2          # type: ignore
import numpy as np  # type: ignore
import rclpy        # type: ignore

from .interface import LineFollowingInterface


class MyLineFollower(LineFollowingInterface):
    """
    Student implementation of line following using SVM.
    
    Detect a green line and steer to stay centered on it using an SVM classifier.
    """

    def __init__(self):
        super().__init__("my_line_follower")
        self._frame_count = 0
        self._lost_count = 0

        # Pre-compute morphological kernel
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # Load SVM model
        model_path = os.path.join(os.path.dirname(__file__), "svm_model.pkl")
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    self._model = pickle.load(f)
                self.get_logger().info("SVM model loaded successfully!")
            except Exception as e:
                self.get_logger().error(f"Failed to load SVM model: {e}")
                self._model = None
        else:
            self.get_logger().warning("SVM model pickle not found! Running in heuristic fallback mode.")
            self._model = None

        # Register camera callback
        self.on_camera_image(self.detect_line)

    def detect_line(self, image: np.ndarray) -> float | None:
        """
        Detect the green line and return steering command.

        Args:
            image: BGR image from camera, shape (720, 1280, 3)

        Returns:
            Steering value in [-1.0, 1.0], or None if line not detected.
        """
        start = time.perf_counter()

        if not hasattr(self, "_frame_count"):
            self._frame_count = 0
        self._frame_count += 1

        # Harness compatibility check for DummySelf instances
        if not hasattr(self, "_kernel"):
            self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        if not hasattr(self, "_lost_count"):
            self._lost_count = 0
        if not hasattr(self, "_model"):
            model_path = os.path.join(os.path.dirname(__file__), "svm_model.pkl")
            if os.path.exists(model_path):
                try:
                    with open(model_path, "rb") as f:
                        self._model = pickle.load(f)
                except Exception:
                    self._model = None
            else:
                self._model = None

        h, w = image.shape[:2]

        # 1. Region of Interest (bottom portion of the frame)
        ROI_BOTTOM_FRACTION = 0.60
        roi_start_row = int(h * (1.0 - ROI_BOTTOM_FRACTION))
        roi = image[roi_start_row:, :]
        roi_h, roi_w = roi.shape[:2]

        # 2. Convert to HSV for robust colour segmentation
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 3. Threshold for green pixels (optimized for dark shadows)
        LOWER_GREEN_HSV = np.array([35, 30, 20])
        UPPER_GREEN_HSV = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, LOWER_GREEN_HSV, UPPER_GREEN_HSV)

        # 4. Morphological cleanup
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)

        # 5. Check if lane is detected (reject if too few green pixels)
        MIN_CONTOUR_AREA = 150
        mask_sum = np.sum(mask > 0)
        if mask_sum < MIN_CONTOUR_AREA:
            return self._handle_no_detection()

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._handle_no_detection()

        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < MIN_CONTOUR_AREA:
            return self._handle_no_detection()

        # Reset lost counter on successful detection
        self._lost_count = 0

        # 6. Model Prediction or Heuristic Fallback
        if self._model is not None:
            # Resize and normalize feature mask
            FEATURE_WIDTH = 30
            FEATURE_HEIGHT = 15
            mask_resized = cv2.resize(mask, (FEATURE_WIDTH, FEATURE_HEIGHT), interpolation=cv2.INTER_AREA)
            features = (mask_resized > 0).astype(np.float32).flatten()

            # Predict probabilities
            try:
                probs = self._model.predict_proba([features])[0]
                # classes are sorted as [-1, 0, 1] -> Left, Straight, Right
                # Expected steering = P(Right) - P(Left)
                steering = float(probs[2] - probs[0])
            except Exception:
                # Prediction fallback
                steering = self._heuristic_steering(largest_contour, roi_h, roi_w)
        else:
            # Heuristic contour fallback
            steering = self._heuristic_steering(largest_contour, roi_h, roi_w)

        # Ensure bounds
        steering = float(np.clip(steering, -1.0, 1.0))

        # Periodic logging and visual notifications
        elapsed_ms = (time.perf_counter() - start) * 1000
        if self._frame_count % 30 == 0:
            mode = "SVM" if self._model is not None else "Heuristic Fallback"
            self.get_logger().info(
                f"[{mode}] steer={steering:+.3f}  frame={self._frame_count}  "
                f"time={elapsed_ms:.1f}ms"
            )

        self.show_notification(f"steer={steering:+.3f}")
        return steering

    def _heuristic_steering(self, contour: np.ndarray, roi_h: int, roi_w: int) -> float:
        """Fallback steering calculation using contour centroid moments."""
        M = cv2.moments(contour)
        if M["m00"] == 0:
            line_x = roi_w / 2.0
        else:
            line_x = M["m10"] / M["m00"]

        image_center_x = roi_w / 2.0
        offset = (line_x - image_center_x) / image_center_x
        # Heuristic gain of 1.0
        return float(offset * 1.0)

    def _handle_no_detection(self) -> None:
        """Handle lost line cases."""
        if not hasattr(self, "_lost_count"):
            self._lost_count = 0
        self._lost_count += 1

        LOST_WARN_FRAMES = 15
        if self._lost_count == LOST_WARN_FRAMES:
            self.get_logger().warning(
                f"Line lost for {LOST_WARN_FRAMES} consecutive frames"
            )
            self.show_warning("Line not detected!")

        return None


def main(args=None):
    """Main entry point for the line follower node."""
    rclpy.init(args=args)
    follower = MyLineFollower()
    try:
        rclpy.spin(follower)
    except KeyboardInterrupt:
        pass
    finally:
        follower.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
