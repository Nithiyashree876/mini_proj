"""
Console-based runner for the Multimodal Identity Recognition System.
Provides real-time face detection, recognition, spoof detection,
and context-aware notification via webcam + optional audio input.

Controls:
  q - Quit
  a - Capture audio for speaker recognition
"""

import sys
import os
import cv2
import numpy as np
import time

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vision.face_detector import FaceDetector
from vision.face_recognizer import FaceRecognizer
from vision.spoof_detector import SpoofDetector
from audio.audio_capture import AudioCapture
from audio.speaker_recognizer import SpeakerRecognizer
from fusion.multimodal_fusion import MultimodalFusion
from language.context_engine import ContextEngine
from language.notification_generator import NotificationGenerator
from utils.database import IdentityDatabase
from utils.logger import EventLogger


def main():
    print("=" * 60)
    print("  Multimodal Identity Recognition System")
    print("  Console Mode")
    print("=" * 60)

    # ── Initialize all modules ────────────────────────────────────
    db = IdentityDatabase()
    face_detector = FaceDetector()
    face_recognizer = FaceRecognizer(db)
    spoof_detector = SpoofDetector()
    audio_capture = AudioCapture()
    speaker_recognizer = SpeakerRecognizer(db)
    fusion = MultimodalFusion()
    context = ContextEngine()
    notifier = NotificationGenerator()
    logger = EventLogger()

    print(f"\n  Database: {db.get_identity_count()} known identities")

    if db.get_identity_count() == 0:
        print("  No identities in database. Run setup_demo.py first!")
        print("    python setup_demo.py")
        return

    # ── Open webcam ───────────────────────────────────────────────
    print("\n  Starting webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  Cannot open webcam")
        return

    print("\nControls:")
    print("  'q' = Quit")
    print("  'a' = Record audio for speaker recognition")
    print("-" * 60)

    frame_count = 0
    process_every_n = 5  # Process every Nth frame for performance
    last_voice_result = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("  Failed to read frame")
                break

            frame_count += 1

            # Process every Nth frame to maintain real-time performance
            if frame_count % process_every_n == 0:
                detections = face_detector.detect_faces(frame)

                identities = []
                for det in detections:
                    # Face recognition
                    face_result = face_recognizer.recognize(det['face_region'])

                    # Eye detection for anti-spoof
                    eyes = face_detector.detect_eyes(det['gray_region'])

                    # Spoof detection
                    spoof_result = spoof_detector.detect_spoof(
                        det['face_region'], det['gray_region'], eyes
                    )

                    # Multimodal fusion (face + latest voice result)
                    fusion_result = fusion.fuse(
                        face_result=face_result,
                        voice_result=last_voice_result,
                        spoof_result=spoof_result
                    )

                    # Update temporal context
                    ctx = context.update(fusion_result)

                    # Generate notification
                    notification = notifier.generate(ctx, fusion_result)

                    # Log the event
                    logger.log_event(
                        event_type=notification['level'],
                        identity=ctx['identity'],
                        confidence=ctx['confidence'],
                        message=notification['message']
                    )

                    # Print significant notifications only
                    if (ctx.get('is_new') or ctx.get('is_returning') or
                            fusion_result['status'] in ['SPOOF', 'CONFLICT', 'UNKNOWN']):
                        print(f"\n{notification['icon']} {notification['message']}")
                        print(f"   {notification['detail']}")

                    identities.append({
                        'name': fusion_result['identity'],
                        'confidence': fusion_result['confidence']
                    })

                # Draw annotated frame
                annotated = face_detector.draw_detections(frame, detections, identities)
            else:
                annotated = frame

            # Show frame
            cv2.imshow('Multimodal Identity Recognition', annotated)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                # ── Audio capture for speaker recognition ─────────
                print("\n  Starting audio capture...")
                try:
                    audio_data = audio_capture.record(duration=3)
                    voice_result = speaker_recognizer.recognize(audio_data)
                    last_voice_result = voice_result
                    print(f"  Speaker: {voice_result['name']} "
                          f"(confidence: {voice_result['confidence']:.2%})")

                    # Re-fuse with audio if we have a recent face
                    if detections:
                        last_face = face_recognizer.recognize(
                            detections[0]['face_region']
                        )
                        fusion_result = fusion.fuse(
                            face_result=last_face,
                            voice_result=voice_result
                        )
                        ctx = context.update(fusion_result)
                        notification = notifier.generate(ctx, fusion_result)
                        print(f"\n{notification['icon']} {notification['message']}")
                        print(f"   {notification['detail']}")

                except Exception as e:
                    print(f"  Audio error: {e}")

    except KeyboardInterrupt:
        print("\n\nStopping...")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # ── Print session summary ─────────────────────────────────
        summary = context.get_session_summary()
        notif_counts = notifier.get_notification_counts()

        print(f"\n{'=' * 60}")
        print(f"  Session Summary")
        print(f"{'=' * 60}")
        print(f"  Duration:           {summary['session_duration']}")
        print(f"  Total detections:   {summary['total_detections']}")
        print(f"  Unique persons:     {summary['unique_persons']}")
        print(f"  Unknown encounters: {summary['unknown_encounters']}")
        print(f"  Notifications:      {notif_counts}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
