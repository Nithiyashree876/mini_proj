"""
Spoof Detection Module.
Implements basic anti-spoofing checks using:
1. Texture analysis (Laplacian variance)
2. Motion analysis (micro-movement detection)
3. Eye presence consistency
4. Color distribution analysis
"""

import cv2
import numpy as np
from utils.config import SPOOF_LBP_THRESHOLD, SPOOF_MOTION_THRESHOLD


class SpoofDetector:
    """Multi-check anti-spoofing detector for face liveness verification."""

    def __init__(self):
        self.previous_frame = None
        self.motion_history = []
        self.blink_history = []
        self.frame_count = 0

    def detect_spoof(self, face_region, gray_face, eyes_detected):
        """Run multiple anti-spoofing checks on a detected face.

        Args:
            face_region: BGR face image
            gray_face: Grayscale face image
            eyes_detected: Array of eye bounding boxes (from FaceDetector.detect_eyes)

        Returns:
            Dict with:
                - is_spoof: bool — whether this appears to be a spoof
                - spoof_confidence: float — probability of spoofing [0, 1]
                - reasons: list of str — human-readable explanations
                - individual_scores: dict of per-check scores
        """
        scores = []
        reasons = []

        # Check 1: Texture analysis
        texture_score = self._check_texture(gray_face)
        scores.append(texture_score)
        if texture_score > 0.5:
            reasons.append("Low texture variance (possible printed photo)")

        # Check 2: Motion analysis
        motion_score = self._check_motion(gray_face)
        scores.append(motion_score)
        if motion_score > 0.5:
            reasons.append("No natural micro-movements detected")

        # Check 3: Eye presence consistency
        eye_score = self._check_eyes(eyes_detected)
        scores.append(eye_score)
        if eye_score > 0.5:
            reasons.append("Eyes not consistently detected")

        # Check 4: Color distribution
        if len(face_region.shape) == 3:
            color_score = self._check_color_distribution(face_region)
            scores.append(color_score)
            if color_score > 0.5:
                reasons.append("Unnatural color distribution")

        # Aggregate spoof score
        avg_score = float(np.mean(scores)) if scores else 0.0
        is_spoof = avg_score > 0.5

        self.frame_count += 1

        return {
            'is_spoof': is_spoof,
            'spoof_confidence': avg_score,
            'reasons': reasons,
            'individual_scores': {
                'texture': texture_score,
                'motion': motion_score,
                'eyes': eye_score
            }
        }

    def _check_texture(self, gray_face):
        """Check face texture using Laplacian variance.

        Real faces have rich, detailed texture. Printed photos or screen
        displays typically have lower texture variance.
        """
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()

        if laplacian_var < SPOOF_LBP_THRESHOLD:
            return 0.8  # High spoof likelihood — very flat image
        elif laplacian_var < SPOOF_LBP_THRESHOLD * 2:
            return 0.3  # Mild concern
        else:
            return 0.1  # Looks real

    def _check_motion(self, gray_face):
        """Check for natural micro-movements between frames.

        Real faces exhibit subtle involuntary movements. A perfectly
        static face suggests a photo or frozen video.
        """
        gray_resized = cv2.resize(gray_face, (64, 64))

        if self.previous_frame is not None:
            diff = cv2.absdiff(self.previous_frame, gray_resized)
            motion = float(diff.sum())
            self.motion_history.append(motion)

            # Keep only the last 30 measurements
            if len(self.motion_history) > 30:
                self.motion_history = self.motion_history[-30:]

            if len(self.motion_history) >= 10:
                motion_std = np.std(self.motion_history[-10:])
                if motion_std < SPOOF_MOTION_THRESHOLD * 0.1:
                    score = 0.7  # Too still — possible photo
                elif motion_std > SPOOF_MOTION_THRESHOLD * 10:
                    score = 0.6  # Too much motion — possible video playback
                else:
                    score = 0.1  # Natural motion range
            else:
                score = 0.3  # Insufficient data
        else:
            score = 0.3  # First frame

        self.previous_frame = gray_resized.copy()
        return score

    def _check_eyes(self, eyes_detected):
        """Check eye detection consistency over time.

        Real faces should have detectable eyes most of the time,
        with occasional misses (blinking). A constant detection or
        constant absence is suspicious.
        """
        has_eyes = len(eyes_detected) >= 1 if eyes_detected is not None and len(eyes_detected) > 0 else False
        self.blink_history.append(has_eyes)

        if len(self.blink_history) > 20:
            self.blink_history = self.blink_history[-20:]

        if len(self.blink_history) >= 10:
            eye_ratio = sum(self.blink_history[-10:]) / 10

            if eye_ratio < 0.3:
                return 0.7  # Eyes rarely detected — suspicious
            elif eye_ratio > 0.95:
                return 0.4  # Eyes always detected (never blinks) — mildly suspicious
            else:
                return 0.1  # Normal blink pattern
        return 0.3  # Insufficient data

    def _check_color_distribution(self, face_region):
        """Check whether the color distribution looks natural for a human face.

        Printed photos and screens often have different saturation profiles
        compared to real skin.
        """
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        sat_std = float(hsv[:, :, 1].std())

        if sat_std < 10:
            return 0.6  # Very uniform saturation — possible print
        elif sat_std > 80:
            return 0.5  # Very high variance — possible screen artifacts
        else:
            return 0.1  # Natural range

    def reset(self):
        """Reset all tracking state (call when switching to a new subject)."""
        self.previous_frame = None
        self.motion_history = []
        self.blink_history = []
        self.frame_count = 0
