"""
Streamlit GUI for the Multimodal Identity Recognition System.
Provides three views:
  1. Live Recognition — webcam feed + real-time notifications
  2. Identity Management — enroll, view, and remove identities
  3. Dashboard — session metrics and event history

Run with: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import time
import sys
import os
from datetime import datetime

# Add project root to path
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
from utils.sms_sender import SMSSender


# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG & CUSTOM STYLING
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Multimodal Identity Recognition",
    page_icon="\U0001f510",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        text-align: center;
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .notification-info {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 4px solid #28a745;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
        font-family: 'Inter', sans-serif;
    }
    .notification-warning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 4px solid #ffc107;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
        font-family: 'Inter', sans-serif;
    }
    .notification-alert {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 4px solid #dc3545;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
        font-family: 'Inter', sans-serif;
    }
    .notification-critical {
        background: linear-gradient(135deg, #f5c6cb 0%, #f1b0b7 100%);
        border-left: 4px solid #721c24;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
        font-family: 'Inter', sans-serif;
    }
    .metric-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 0.75rem;
        padding: 1.25rem;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .metric-box h3 {
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #495057;
        margin: 0;
    }
    .metric-box p {
        font-family: 'Inter', sans-serif;
        color: #868e96;
        font-size: 0.85rem;
        margin: 0.25rem 0 0 0;
    }
    .stButton>button {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SYSTEM INITIALIZATION (cached across reruns)
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def init_system():
    """Initialize all system components (cached once per session)."""
    db = IdentityDatabase()
    return {
        'db': db,
        'face_detector': FaceDetector(),
        'face_recognizer': FaceRecognizer(db),
        'spoof_detector': SpoofDetector(),
        'audio_capture': AudioCapture(),
        'speaker_recognizer': SpeakerRecognizer(db),
        'fusion': MultimodalFusion(),
        'context': ContextEngine(),
        'notifier': NotificationGenerator(),
        'logger': EventLogger(),
        'sms': SMSSender()
    }


# ══════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════
def main():
    st.markdown(
        '<div class="main-header">Multimodal Identity Recognition System</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">Real-time face + voice identification with context-aware alerts</div>',
        unsafe_allow_html=True
    )

    system = init_system()
    db = system['db']

    # ── Sidebar ───────────────────────────────────────────────────
    st.sidebar.title("Controls")

    mode = st.sidebar.radio(
        "Mode",
        ["Live Recognition", "Identity Management", "Dashboard"],
        format_func=lambda x: {
            "Live Recognition": "\U0001f3a5 Live Recognition",
            "Identity Management": "\U0001f464 Identity Management",
            "Dashboard": "\U0001f4ca Dashboard"
        }[x]
    )

    st.sidebar.divider()
    st.sidebar.metric("Known Identities", db.get_identity_count())
    st.sidebar.metric("Session Events", system['logger'].get_event_count())

    # ── Route to page ─────────────────────────────────────────────
    if mode == "Live Recognition":
        live_recognition_page(system)
    elif mode == "Identity Management":
        identity_management_page(system)
    elif mode == "Dashboard":
        dashboard_page(system)


# ══════════════════════════════════════════════════════════════════
# LIVE RECOGNITION PAGE
# ══════════════════════════════════════════════════════════════════
def live_recognition_page(system):
    st.subheader("Live Recognition")

    if system['db'].get_identity_count() == 0:
        st.warning(
            "No identities enrolled yet! Go to **Identity Management** > "
            "**Enroll New** to add people, or click the button below to "
            "generate synthetic demo data."
        )
        if st.button("Generate Demo Identities"):
            _generate_demo_data(system)
            st.rerun()
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.info(
            "Click **Start Camera** to begin face detection. "
            "Press **Stop** to end. Uses your default webcam."
        )

        run = st.checkbox("Start Camera", value=False, key="cam_toggle")
        camera_placeholder = st.empty()

        if run:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot access webcam. Make sure your camera is connected and not in use.")
                return

            frame_count = 0
            last_detections = []
            last_identities_list = []
            stop_requested = st.button("Stop Camera")

            while run and not stop_requested:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to read frame from camera.")
                    break

                frame_count += 1

                if frame_count % 3 == 0:
                    detections = system['face_detector'].detect_faces(frame)

                    identities_list = []
                    for det in detections:
                        face_result = system['face_recognizer'].recognize(frame, det['yunet_face'])
                        eyes = system['face_detector'].detect_eyes(det['gray_region'])
                        spoof_result = system['spoof_detector'].detect_spoof(
                            det['face_region'], det['gray_region'], eyes
                        )

                        fusion_result = system['fusion'].fuse(
                            face_result=face_result,
                            spoof_result=spoof_result
                        )

                        ctx = system['context'].update(fusion_result)
                        notification = system['notifier'].generate(ctx, fusion_result)

                        system['logger'].log_event(
                            event_type=notification['level'],
                            identity=ctx['identity'],
                            confidence=ctx['confidence'],
                            message=notification['message']
                        )

                        # Trigger SMS if it's the first time we see this person this session
                        if ctx.get('is_new') and fusion_result.get('person_id'):
                            person_info = system['db'].get_identity(fusion_result['person_id'])
                            if person_info and person_info.get('metadata', {}).get('phone'):
                                msg = f"Security Alert: {person_info['name']}, your face was successfully recognized by the system at {datetime.now().strftime('%I:%M %p')}!"
                                system['sms'].send_sms(person_info['metadata']['phone'], msg)
                                st.toast(f"📱 SMS Notification triggered for {person_info['name']}!")

                        identities_list.append({
                            'name': fusion_result['identity'],
                            'confidence': fusion_result['confidence']
                        })
                        
                    last_detections = detections
                    last_identities_list = identities_list

                    annotated = system['face_detector'].draw_detections(
                        frame, detections, identities_list
                    )
                else:
                    annotated = system['face_detector'].draw_detections(
                        frame, last_detections, last_identities_list
                    )

                frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                time.sleep(0.03)

            cap.release()

    with col2:
        st.subheader("Notifications")
        _render_notifications(system)

        st.divider()

        st.subheader("Audio Recognition")
        
        tab_mic, tab_sys = st.tabs(["Browser Mic", "System Mic (Fallback)"])
        
        with tab_mic:
            audio_val = st.audio_input("Record your voice to test Speaker Recognition")
            
            if audio_val is not None:
                try:
                    import soundfile as sf
                    audio_data, sr = sf.read(audio_val)
                    
                    # Convert to mono if it captured stereo
                    if len(audio_data.shape) > 1:
                        audio_data = audio_data.mean(axis=1)
                    
                    with st.spinner("Processing voice profile..."):
                        voice_result = system['speaker_recognizer'].recognize(audio_data, sr=sr)
                    
                    if voice_result['name'] != 'Unknown':
                        st.success(
                            f"Recognized Speaker: **{voice_result['name']}** "
                            f"({voice_result['confidence']:.0%} confidence)"
                        )
                    else:
                        st.warning(
                            f"Speaker not recognized "
                            f"(best match: {voice_result['confidence']:.0%})"
                        )
                except Exception as e:
                    st.error(f"Error processing audio: {e}")
                    
        with tab_sys:
            st.info("If the browser recording doesn't capture audio (00:00 length), use your system's native microphone directly. It records for 3 seconds.")
            if st.button("🎤 Record 3s via System Mic"):
                with st.spinner("Recording... Please speak now!"):
                    try:
                        audio_data = system['audio_capture'].record(duration=3)
                        voice_result = system['speaker_recognizer'].recognize(audio_data)
                        
                        if voice_result['name'] != 'Unknown':
                            st.success(
                                f"Recognized Speaker: **{voice_result['name']}** "
                                f"({voice_result['confidence']:.0%} confidence)"
                            )
                        else:
                            st.warning(
                                f"Speaker not recognized "
                                f"(best match: {voice_result['confidence']:.0%})"
                            )
                    except Exception as e:
                        st.error(f"System Microphone error: {e}")


# ══════════════════════════════════════════════════════════════════
# IDENTITY MANAGEMENT PAGE
# ══════════════════════════════════════════════════════════════════
def identity_management_page(system):
    st.subheader("Identity Management")

    db = system['db']

    tab1, tab2, tab3 = st.tabs(["View Identities", "Enroll New", "Remove"])

    # ── View ──────────────────────────────────────────────────────
    with tab1:
        identities = db.get_all_identities()
        if identities:
            for pid, info in identities.items():
                with st.expander(f"\U0001f464 {info['name']} ({pid})"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Face Samples", len(info.get('face_embeddings', [])))
                    c2.metric("Voice Samples", len(info.get('speaker_embeddings', [])))
                    c3.metric("Created", info.get('created_at', 'N/A')[:10])
        else:
            st.warning("No identities enrolled yet. Use the **Enroll New** tab.")

    # ── Enroll ────────────────────────────────────────────────────
    with tab2:
        st.write("### Enroll via Webcam")
        name = st.text_input("Person's Name", key="enroll_name")
        phone = st.text_input("Phone Number (for SMS)", key="enroll_phone", help="+1234567890 format")
        num_samples = st.slider("Number of face samples", 1, 10, 5)

        if st.button("Start Enrollment") and name:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot access webcam.")
            else:
                progress = st.progress(0)
                status_text = st.empty()
                captures = 0
                pid = None

                while captures < num_samples:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    detections = system['face_detector'].detect_faces(frame)
                    if detections:
                        embedding = system['face_recognizer'].extract_embedding(
                            frame, detections[0]['yunet_face']
                        )
                        if pid is None:
                            pid = db.add_identity(name, face_embedding=embedding, metadata={'phone': phone})
                        else:
                            db.add_face_embedding(pid, embedding)

                        captures += 1
                        progress.progress(captures / num_samples)
                        status_text.text(f"Captured {captures}/{num_samples} samples")
                        time.sleep(0.5)

                cap.release()

                if captures > 0 and pid:
                    # Add synthetic speaker profile
                    _add_synthetic_speaker(pid, name, db, system['speaker_recognizer'])
                    st.success(f"Enrolled **{name}** with {captures} face samples!")
                    st.rerun()
                else:
                    st.error("No faces detected during enrollment.")

        st.divider()

        st.write("### Quick Demo Setup")
        if st.button("Generate Synthetic Identities"):
            _generate_demo_data(system)
            st.rerun()

    # ── Remove ────────────────────────────────────────────────────
    with tab3:
        identities = db.get_all_identities()
        if identities:
            pid_to_remove = st.selectbox(
                "Select identity to remove",
                options=list(identities.keys()),
                format_func=lambda x: f"{x}: {identities[x]['name']}"
            )
            if st.button("Remove Identity", type="primary"):
                db.remove_identity(pid_to_remove)
                st.success(f"Removed {pid_to_remove}")
                st.rerun()
        else:
            st.info("No identities to remove.")


# ══════════════════════════════════════════════════════════════════
# DASHBOARD PAGE
# ══════════════════════════════════════════════════════════════════
def dashboard_page(system):
    st.subheader("System Dashboard")

    summary = system['context'].get_session_summary()
    notif_counts = system['notifier'].get_notification_counts()

    # ── Metrics row ───────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f'<div class="metric-box"><h3>{summary["total_detections"]}</h3>'
        f'<p>Total Detections</p></div>',
        unsafe_allow_html=True
    )
    c2.markdown(
        f'<div class="metric-box"><h3>{summary["unique_persons"]}</h3>'
        f'<p>Unique Persons</p></div>',
        unsafe_allow_html=True
    )
    c3.markdown(
        f'<div class="metric-box"><h3>{summary["unknown_encounters"]}</h3>'
        f'<p>Unknown Encounters</p></div>',
        unsafe_allow_html=True
    )
    duration_str = str(summary['session_duration']).split('.')[0]
    c4.markdown(
        f'<div class="metric-box"><h3>{duration_str}</h3>'
        f'<p>Session Duration</p></div>',
        unsafe_allow_html=True
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Notification Summary")
        for level, count in notif_counts.items():
            icon = {
                'INFO': '\u2705', 'WARNING': '\u26a0\ufe0f',
                'ALERT': '\U0001f514', 'CRITICAL': '\U0001f6a8'
            }.get(level, '\U0001f4e2')
            st.write(f"{icon} **{level}**: {count}")

    with col2:
        st.subheader("Active Persons")
        active = system['context'].get_active_persons()
        if active:
            for pid, count in active.items():
                info = system['db'].get_identity(pid)
                name = info['name'] if info else pid
                st.write(f"\U0001f464 **{name}**: {count} detections in last 60s")
        else:
            st.info("No active persons detected.")

    st.divider()

    st.subheader("Recent Events")
    events = system['logger'].get_recent_events(20)
    if events:
        for event in reversed(events):
            ts = event.get('timestamp', '')[:19]
            st.text(f"[{ts}] [{event.get('type', 'INFO'):>8}] {event.get('message', '')}")
    else:
        st.info("No events recorded yet. Start live recognition to generate events.")


# ══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════
def _render_notifications(system):
    """Render recent notifications with styled cards."""
    notifications = system['notifier'].get_recent_notifications(10)
    if notifications:
        for notif in reversed(notifications):
            level = notif.get('level', 'INFO').lower()
            css_class = f'notification-{level}'
            st.markdown(
                f'<div class="{css_class}">'
                f'<strong>{notif.get("icon", "")} {notif.get("message", "")}</strong><br>'
                f'<small>{notif.get("detail", "")}</small>'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.info("No notifications yet. Start the camera to begin detecting.")


def _generate_demo_data(system):
    """Generate synthetic identities for demo purposes."""
    from setup_demo import enroll_synthetic_face, enroll_synthetic_speaker

    db = system['db']
    names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve']

    progress = st.progress(0)
    for i, name in enumerate(names):
        pid = enroll_synthetic_face(name, db, system['face_recognizer'])
        if pid:
            enroll_synthetic_speaker(pid, name, db, system['speaker_recognizer'])
        progress.progress((i + 1) / len(names))

    st.success(f"Created {len(names)} synthetic identities!")


def _add_synthetic_speaker(pid, name, db, speaker_recognizer):
    """Add synthetic speaker profile to an existing identity."""
    base_freq = 100 + (hash(name) % 150)
    for i in range(3):
        audio = AudioCapture.generate_synthetic_audio(
            duration=3,
            base_freq=base_freq + i * 5,
            seed=hash(name + str(i)) % (2**31)
        )
        embedding = speaker_recognizer.extract_embedding(audio)
        db.add_speaker_embedding(pid, embedding)


if __name__ == "__main__":
    main()
