"""
Centralized configuration for the Multimodal Identity Recognition System.
All tunable parameters are defined here for easy adjustment.
"""

import os

# ============================================================
# DIRECTORY PATHS
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
FACES_DIR = os.path.join(DATA_DIR, "known_faces")
SPEAKERS_DIR = os.path.join(DATA_DIR, "known_speakers")
LOGS_DIR = os.path.join(DATA_DIR, "logs")
DB_PATH = os.path.join(DATA_DIR, "identity_db.json")

# ============================================================
# VISION MODULE SETTINGS
# ============================================================
FACE_DETECTION_SCALE = 1.3          # Scale factor for Haar cascade
FACE_DETECTION_MIN_NEIGHBORS = 5    # Min neighbors for detection robustness
FACE_MIN_SIZE = (60, 60)            # Minimum face size in pixels
FACE_EMBEDDING_SIZE = (100, 100)    # Face image resize for embedding extraction
FACE_RECOGNITION_THRESHOLD = 0.72   # Minimum cosine similarity to accept match

# ============================================================
# AUDIO MODULE SETTINGS
# ============================================================
AUDIO_SAMPLE_RATE = 16000            # Sample rate in Hz
AUDIO_DURATION = 3                   # Default recording duration in seconds
AUDIO_N_MFCC = 20                    # Number of MFCC coefficients
SPEAKER_RECOGNITION_THRESHOLD = 0.70 # Minimum cosine similarity for speaker match

# ============================================================
# FUSION SETTINGS
# ============================================================
FUSION_FACE_WEIGHT = 0.6            # Weight for face modality in fusion
FUSION_VOICE_WEIGHT = 0.4           # Weight for voice modality in fusion
FUSION_CONFLICT_THRESHOLD = 0.3     # Threshold for flagging modality conflict

# ============================================================
# SPOOF DETECTION SETTINGS
# ============================================================
SPOOF_LBP_THRESHOLD = 50            # Laplacian variance threshold for texture check
SPOOF_MOTION_THRESHOLD = 5000       # Motion magnitude threshold

# ============================================================
# NOTIFICATION / LOGGING
# ============================================================
NOTIFICATION_LOG_FILE = os.path.join(LOGS_DIR, "events.log")

# ============================================================
# Create required directories on import
# ============================================================
for _dir in [DATA_DIR, FACES_DIR, SPEAKERS_DIR, LOGS_DIR]:
    os.makedirs(_dir, exist_ok=True)
