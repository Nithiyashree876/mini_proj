"""
Face Detection Module.
Uses OpenCV Haar cascades for real-time face detection from video frames.
Provides bounding boxes, cropped face regions, and annotation drawing.
"""

import cv2
import numpy as np
from utils.config import (
    FACE_DETECTION_SCALE,
    FACE_DETECTION_MIN_NEIGHBORS,
    FACE_MIN_SIZE
)


class FaceDetector:
    """Real-time face detector using OpenCV Haar cascades."""

    def __init__(self):
        import os
        # Load pre-trained deep learning cascades (YuNet)
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'yunet.onnx'))
        try:
            self.detector = cv2.FaceDetectorYN.create(
                model=model_path,
                config="",
                input_size=(320, 320),
                score_threshold=0.85,
                nms_threshold=0.3,
                top_k=5000
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load YuNet model: {e}. Ensure models from opencv zoo are downloaded.")
            
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )

    def detect_faces(self, frame):
        """Detect faces in a BGR video frame using YuNet DL model.

        Args:
            frame: BGR image (numpy array from OpenCV)

        Returns:
            List of detection dicts, each containing:
                - bbox: (x, y, w, h) bounding box
                - face_region: cropped BGR face image
                - gray_region: cropped grayscale face image
                - yunet_face: The raw 15-float array needed for SFace alignment
        """
        raw_height, raw_width = frame.shape[:2]
        self.detector.setInputSize((raw_width, raw_height))
        ret, faces = self.detector.detect(frame)
        
        results = []
        if faces is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            for face in faces:
                x, y, w, h = face[:4]
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                # Ensure bbox stays within bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, raw_width - x)
                h = min(h, raw_height - y)
                
                if w > 0 and h > 0:
                    face_region = frame[y:y+h, x:x+w]
                    gray_region = gray[y:y+h, x:x+w]
                    
                    results.append({
                        'bbox': (x, y, w, h),
                        'face_region': face_region,
                        'gray_region': gray_region,
                        'yunet_face': face
                    })

        return results

    def detect_eyes(self, gray_face):
        """Detect eyes within a grayscale face region.

        Used for liveness / anti-spoof checking (blink detection).

        Args:
            gray_face: Grayscale face image

        Returns:
            Array of (x, y, w, h) bounding boxes for detected eyes
        """
        eyes = self.eye_cascade.detectMultiScale(
            gray_face,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        return eyes

    def draw_detections(self, frame, detections, identities=None):
        """Draw bounding boxes and identity labels on a frame.

        Args:
            frame: Original BGR frame
            detections: List of detection dicts from detect_faces()
            identities: Optional list of identity dicts with 'name' and 'confidence'

        Returns:
            Annotated copy of the frame
        """
        annotated = frame.copy()

        for i, det in enumerate(detections):
            x, y, w, h = det['bbox']

            # Determine color and label based on identity
            if identities and i < len(identities):
                identity = identities[i]
                name = identity.get('name', 'Unknown')
                confidence = identity.get('confidence', 0)

                if name == 'Unknown':
                    color = (0, 0, 255)      # Red for unknown
                elif name == 'Spoof Detected':
                    color = (0, 0, 200)      # Dark red for spoof
                else:
                    color = (0, 255, 0)      # Green for recognized

                label = f"{name} ({confidence:.0%})"
            else:
                color = (255, 200, 0)        # Cyan-ish for detecting
                label = "Detecting..."

            # Draw bounding box with rounded corners effect
            thickness = 2
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, thickness)

            # Corner accents for a modern look
            corner_len = min(20, w // 4, h // 4)
            cv2.line(annotated, (x, y), (x + corner_len, y), color, thickness + 2)
            cv2.line(annotated, (x, y), (x, y + corner_len), color, thickness + 2)
            cv2.line(annotated, (x + w, y), (x + w - corner_len, y), color, thickness + 2)
            cv2.line(annotated, (x + w, y), (x + w, y + corner_len), color, thickness + 2)
            cv2.line(annotated, (x, y + h), (x + corner_len, y + h), color, thickness + 2)
            cv2.line(annotated, (x, y + h), (x, y + h - corner_len), color, thickness + 2)
            cv2.line(annotated, (x + w, y + h), (x + w - corner_len, y + h), color, thickness + 2)
            cv2.line(annotated, (x + w, y + h), (x + w, y + h - corner_len), color, thickness + 2)

            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(
                annotated,
                (x, y - label_size[1] - 10),
                (x + label_size[0] + 6, y),
                color, -1
            )
            cv2.putText(
                annotated, label, (x + 3, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

        # Draw frame info
        cv2.putText(
            annotated,
            f"Faces: {len(detections)}",
            (10, annotated.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )

        return annotated
