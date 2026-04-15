"""
Demo Setup Script.
Enrolls identities into the database using either:
1. Webcam face capture (real faces)
2. Synthetic data generation (no hardware needed)
3. Mixed mode (webcam faces + synthetic audio)

Run this before main.py or app.py to populate the identity database.
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
from audio.audio_capture import AudioCapture
from audio.speaker_recognizer import SpeakerRecognizer
from utils.database import IdentityDatabase


def enroll_face_from_webcam(name, db, detector, recognizer, num_samples=5):
    """Capture face samples from webcam and enroll as a new identity.

    Args:
        name: Person's name
        db: IdentityDatabase instance
        detector: FaceDetector instance
        recognizer: FaceRecognizer instance
        num_samples: Number of face samples to capture

    Returns:
        Person ID string, or None if enrollment failed
    """
    print(f"\n  Enrolling face for '{name}'...")
    print(f"   Position your face in the center of the camera.")
    print(f"   Press SPACE to capture ({num_samples} samples needed), 'q' to skip.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("   Cannot open webcam. Using synthetic face data instead.")
        return enroll_synthetic_face(name, db, recognizer)

    pid = None
    captures = 0

    while captures < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect_faces(frame)

        # Draw face bounding boxes
        for det in detections:
            x, y, w, h = det['bbox']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(
            frame, f"Enrolling: {name} ({captures}/{num_samples})",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        cv2.putText(
            frame, "Press SPACE to capture, 'q' to skip",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )

        cv2.imshow('Face Enrollment', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and len(detections) > 0:
            embedding = recognizer.extract_embedding(frame, detections[0]['yunet_face'])

            if pid is None:
                pid = db.add_identity(name, face_embedding=embedding)
                print(f"   Created identity {pid}")
            else:
                db.add_face_embedding(pid, embedding)

            captures += 1
            print(f"   Captured sample {captures}/{num_samples}")
            time.sleep(0.3)
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if captures > 0:
        print(f"   Enrolled {name} with {captures} face samples")
        return pid
    else:
        print(f"   No faces captured for {name}")
        return None


def enroll_synthetic_face(name, db, recognizer, num_samples=5):
    """Generate synthetic face images and enroll as a new identity.

    Creates deterministic synthetic faces based on the person's name,
    so re-running with the same name produces the same embeddings.

    Args:
        name: Person's name
        db: IdentityDatabase instance
        recognizer: FaceRecognizer instance
        num_samples: Number of synthetic samples

    Returns:
        Person ID string
    """
    print(f"   Generating synthetic face data for '{name}'...")

    pid = None
    np.random.seed(hash(name) % (2**31))

    for i in range(num_samples):
        # Generate 128D random embedding directly since DL detectors ignore cartoon pixel faces
        np.random.seed((hash(name) + i) % (2**31))
        embedding = np.random.normal(size=128).astype(np.float32)
        embedding /= np.linalg.norm(embedding)



        if pid is None:
            pid = db.add_identity(name, face_embedding=embedding)
        else:
            db.add_face_embedding(pid, embedding)

    print(f"   Enrolled {name} with {num_samples} synthetic face samples")
    return pid


def enroll_synthetic_speaker(pid, name, db, speaker_recognizer, num_samples=3):
    """Generate synthetic speaker audio and add to an existing identity.

    Each person gets a unique base frequency derived from their name,
    creating distinguishable speaker profiles.

    Args:
        pid: Person ID to add speaker data to
        name: Person's name (used to derive unique voice characteristics)
        db: IdentityDatabase instance
        speaker_recognizer: SpeakerRecognizer instance
        num_samples: Number of audio samples
    """
    print(f"   Generating synthetic speaker profile for '{name}'...")

    base_freq = 100 + (hash(name) % 150)

    for i in range(num_samples):
        audio = AudioCapture.generate_synthetic_audio(
            duration=3,
            base_freq=base_freq + i * 5,
            seed=hash(name + str(i)) % (2**31)
        )
        embedding = speaker_recognizer.extract_embedding(audio)
        db.add_speaker_embedding(pid, embedding)

    print(f"   Enrolled {num_samples} speaker samples for {name}")


def main():
    """Interactive setup wizard."""
    print("=" * 60)
    print("  Identity Recognition System - Demo Setup")
    print("=" * 60)

    db = IdentityDatabase()
    detector = FaceDetector()
    face_recognizer = FaceRecognizer(db)
    speaker_recognizer = SpeakerRecognizer(db)

    print(f"\nCurrent database: {db.get_identity_count()} identities")

    print("\nChoose setup mode:")
    print("  1. Webcam enrollment (capture real faces)")
    print("  2. Synthetic demo data (no webcam needed)")
    print("  3. Mixed (webcam faces + synthetic audio)")

    choice = input("\nEnter choice (1/2/3) [default: 2]: ").strip()
    if choice == '':
        choice = '2'

    if choice == '1':
        # ── Webcam enrollment ─────────────────────────────────────
        n = input("How many people to enroll? [default: 3]: ").strip()
        n = int(n) if n else 3

        for i in range(n):
            name = input(f"\nEnter name for person {i + 1}: ").strip()
            if not name:
                name = f"Person_{i + 1}"
            pid = enroll_face_from_webcam(name, db, detector, face_recognizer)
            if pid:
                enroll_synthetic_speaker(pid, name, db, speaker_recognizer)

    elif choice == '2':
        # ── Full synthetic demo ───────────────────────────────────
        names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve']
        print(f"\nGenerating synthetic data for: {', '.join(names)}")

        for name in names:
            pid = enroll_synthetic_face(name, db, face_recognizer)
            if pid:
                enroll_synthetic_speaker(pid, name, db, speaker_recognizer)

    elif choice == '3':
        # ── Mixed mode ────────────────────────────────────────────
        n = input("How many people to enroll? [default: 3]: ").strip()
        n = int(n) if n else 3

        for i in range(n):
            name = input(f"\nEnter name for person {i + 1}: ").strip()
            if not name:
                name = f"Person_{i + 1}"
            pid = enroll_face_from_webcam(name, db, detector, face_recognizer)
            if pid:
                enroll_synthetic_speaker(pid, name, db, speaker_recognizer)

    else:
        print("Invalid choice. Exiting.")
        return

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Setup Complete!")
    print(f"  Total identities: {db.get_identity_count()}")
    print(f"{'=' * 60}")
    for pid, info in db.get_all_identities().items():
        faces = len(info.get('face_embeddings', []))
        voices = len(info.get('speaker_embeddings', []))
        print(f"  {pid}: {info['name']} ({faces} face, {voices} voice samples)")
    print(f"{'=' * 60}")
    print(f"\nNext steps:")
    print(f"  Console mode:  python main.py")
    print(f"  GUI mode:      streamlit run app.py")


if __name__ == "__main__":
    main()
