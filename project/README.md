# Multimodal Intelligent Identity Recognition System

A real-time multimodal identity recognition system that combines **face recognition**, **speaker recognition**, **intelligent fusion**, and **context-aware notification generation**.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Streamlit Web UI                           │
│         (Live Camera Feed + Audio Controls + Alerts)          │
├─────────────┬─────────────┬─────────────┬────────────────────┤
│   Vision    │   Audio     │  Fusion     │    Language         │
│   Module    │   Module    │  Engine     │    Engine           │
│             │             │             │                     │
│ • Detect    │ • Capture   │ • Weighted  │ • Context tracking  │
│ • Embed     │ • MFCC      │   scoring   │ • Smart alerts      │
│ • Match     │ • Match     │ • Conflict  │ • Temporal reason   │
│ • Anti-spoof│ • Confidence│   handling  │ • Spoof warnings    │
├─────────────┴─────────────┴─────────────┴────────────────────┤
│                   Identity Database                           │
│         (Face Embeddings + Speaker Embeddings + Metadata)     │
├──────────────────────────────────────────────────────────────┤
│                   Event Logger & Notifier                     │
│           (Timestamped logs, file output, alerts)             │
└──────────────────────────────────────────────────────────────┘
```

## Features

- **Face Detection** — OpenCV Haar cascade with real-time bounding boxes
- **Face Recognition** — Multi-feature embeddings (pixel + LBP + HOG) with cosine similarity
- **Speaker Recognition** — MFCC-based speaker embeddings with statistical summarization
- **Anti-Spoofing** — Texture analysis, motion detection, eye consistency, color distribution
- **Multimodal Fusion** — Confidence-weighted scoring with conflict handling
- **Context-Aware Notifications** — Temporal tracking, return-visit detection, natural language alerts
- **Privacy-Aware** — Stores only embeddings, never raw images or audio
- **Extensible Architecture** — Modular design, easy to swap in advanced models

## Project Structure

```
project/
├── vision/
│   ├── face_detector.py        # Haar cascade face detection
│   ├── face_recognizer.py      # Face embedding extraction & matching
│   └── spoof_detector.py       # Anti-spoofing checks
├── audio/
│   ├── audio_capture.py        # Microphone capture & synthetic audio
│   └── speaker_recognizer.py   # MFCC-based speaker recognition
├── fusion/
│   └── multimodal_fusion.py    # Confidence-weighted identity fusion
├── language/
│   ├── context_engine.py       # Temporal context tracking
│   └── notification_generator.py  # Context-aware notification generation
├── utils/
│   ├── config.py               # Centralized configuration
│   ├── database.py             # JSON-backed identity database
│   └── logger.py               # Event logging
├── data/                        # Auto-created: embeddings, logs
├── main.py                      # Console-mode runner
├── app.py                       # Streamlit GUI
├── setup_demo.py                # Demo data generator
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Setup Instructions

### 1. Prerequisites
- Python 3.8 or higher
- Webcam (optional — synthetic mode works without one)
- Microphone (optional — for speaker recognition)

### 2. Install Dependencies

```bash
cd project
pip install -r requirements.txt
```

### 3. Set Up Demo Identities

**Option A: Synthetic data (no webcam needed)**
```bash
python setup_demo.py
# Choose option 2 when prompted
```

**Option B: Webcam enrollment**
```bash
python setup_demo.py
# Choose option 1, then follow instructions to capture faces
```

### 4. Run the System

**Console mode (with OpenCV window):**
```bash
python main.py
```
- Press `q` to quit
- Press `a` to capture audio for speaker recognition

**GUI mode (Streamlit web app):**
```bash
streamlit run app.py
```
- Opens in your browser at http://localhost:8501
- Use the sidebar to switch between Live Recognition, Identity Management, and Dashboard

## Example Outputs

### Console Mode
```
══════════════════════════════════════════════════════════════
  Multimodal Identity Recognition System
  Console Mode
══════════════════════════════════════════════════════════════

  Database: 5 known identities

🔍 Alice detected at 06:45:12 PM via face recognition only
   Good evening, Alice! Identified via face recognition only
   with 78% confidence. Note: single-modality identification —
   additional verification recommended.

🆕 Unknown person detected at 06:45:38 PM
   An unrecognized individual was detected with low confidence (12%).
   Total unknown encounters this session: 1.

══════════════════════════════════════════════════════════════
  Session Summary
══════════════════════════════════════════════════════════════
  Duration:           0:02:15
  Total detections:   47
  Unique persons:     2
  Unknown encounters: 5
══════════════════════════════════════════════════════════════
```

### Notification Types
| Status | Description | Example |
|--------|-------------|---------|
| CONFIRMED | Face + Voice agree | "Alice entered at 10:32 AM" |
| IDENTIFIED | Single modality match | "Bob detected via face recognition" |
| UNKNOWN | No match found | "Unknown person detected" |
| CONFLICT | Face ≠ Voice | "Face says Alice, voice says Bob" |
| SPOOF | Anti-spoofing triggered | "Possible spoofing attempt" |

## Configuration

All tunable parameters are in `utils/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FACE_RECOGNITION_THRESHOLD` | 0.55 | Min cosine similarity for face match |
| `SPEAKER_RECOGNITION_THRESHOLD` | 0.70 | Min cosine similarity for speaker match |
| `FUSION_FACE_WEIGHT` | 0.6 | Weight for face in multimodal fusion |
| `FUSION_VOICE_WEIGHT` | 0.4 | Weight for voice in multimodal fusion |
| `SPOOF_LBP_THRESHOLD` | 50 | Laplacian variance for texture check |
| `AUDIO_DURATION` | 3 | Default recording duration (seconds) |

## Suggestions for Improvements

1. **Deep Learning Embeddings** — Replace HOG/LBP with FaceNet or ArcFace for production-grade accuracy
2. **Neural Speaker Verification** — Use ECAPA-TDNN or x-vectors instead of MFCC statistics
3. **Transformer Fusion** — Replace weighted scoring with a learned attention-based fusion network
4. **LLM Integration** — Connect to GPT/Gemini API for truly dynamic notification generation
5. **Database Backend** — Migrate from JSON to SQLite or PostgreSQL for scale
6. **GPU Acceleration** — Use CUDA-enabled OpenCV or ONNX Runtime for faster inference
7. **Distributed Processing** — Separate capture, recognition, and notification into microservices
8. **Advanced Anti-Spoofing** — Add depth estimation, infrared analysis, or challenge-response

## License

This project is provided as an academic prototype for research and educational purposes.
