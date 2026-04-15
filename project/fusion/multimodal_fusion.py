"""
Multimodal Fusion Module.
Combines face recognition and speaker recognition results using
confidence-weighted scoring with intelligent conflict handling.
"""

import numpy as np
from utils.config import (
    FUSION_FACE_WEIGHT,
    FUSION_VOICE_WEIGHT,
    FUSION_CONFLICT_THRESHOLD
)


class MultimodalFusion:
    """Confidence-weighted fusion of face and voice recognition results."""

    def __init__(self):
        self.face_weight = FUSION_FACE_WEIGHT
        self.voice_weight = FUSION_VOICE_WEIGHT
        self.conflict_threshold = FUSION_CONFLICT_THRESHOLD

    def fuse(self, face_result=None, voice_result=None, spoof_result=None):
        """Fuse face and voice recognition results into a single identity decision.

        Handles five scenarios:
        1. Spoof detected → immediate alert
        2. Neither modality recognized → Unknown
        3. Only face recognized → use face
        4. Only voice recognized → use voice
        5. Both recognized:
           a. Agreement → boosted confidence
           b. Conflict → penalized confidence + warning

        Args:
            face_result: Dict from FaceRecognizer.recognize() (or None)
            voice_result: Dict from SpeakerRecognizer.recognize() (or None)
            spoof_result: Dict from SpoofDetector.detect_spoof() (or None)

        Returns:
            Dict with: identity, person_id, confidence, modality, status, details
        """
        # ── Priority 1: Spoof Alert ─────────────────────────────────
        if spoof_result and spoof_result.get('is_spoof', False):
            return {
                'identity': 'Spoof Detected',
                'person_id': None,
                'confidence': spoof_result.get('spoof_confidence', 0.0),
                'modality': 'spoof_alert',
                'status': 'SPOOF',
                'details': {
                    'face': face_result,
                    'voice': voice_result,
                    'spoof': spoof_result
                }
            }

        # Determine which modalities produced a positive match
        has_face = (
            face_result is not None
            and face_result.get('name') != 'Unknown'
            and face_result.get('confidence', 0) > 0
        )
        has_voice = (
            voice_result is not None
            and voice_result.get('name') != 'Unknown'
            and voice_result.get('confidence', 0) > 0
        )

        # ── Case 1: Neither modality recognized ─────────────────────
        if not has_face and not has_voice:
            face_conf = face_result.get('confidence', 0) if face_result else 0
            voice_conf = voice_result.get('confidence', 0) if voice_result else 0
            return {
                'identity': 'Unknown',
                'person_id': None,
                'confidence': max(face_conf, voice_conf),
                'modality': 'none',
                'status': 'UNKNOWN',
                'details': {
                    'face': face_result,
                    'voice': voice_result
                }
            }

        # ── Case 2: Only face recognized ─────────────────────────────
        if has_face and not has_voice:
            return {
                'identity': face_result['name'],
                'person_id': face_result.get('person_id'),
                'confidence': face_result['confidence'],
                'modality': 'face_only',
                'status': 'IDENTIFIED',
                'details': {
                    'face': face_result,
                    'voice': voice_result
                }
            }

        # ── Case 3: Only voice recognized ────────────────────────────
        if has_voice and not has_face:
            return {
                'identity': voice_result['name'],
                'person_id': voice_result.get('person_id'),
                'confidence': voice_result['confidence'],
                'modality': 'voice_only',
                'status': 'IDENTIFIED',
                'details': {
                    'face': face_result,
                    'voice': voice_result
                }
            }

        # ── Case 4: Both modalities recognized ──────────────────────
        face_name = face_result['name']
        voice_name = voice_result['name']
        face_conf = face_result['confidence']
        voice_conf = voice_result['confidence']

        if face_name == voice_name:
            # ── 4a: Agreement → boost confidence ─────────────────────
            fused_confidence = (
                self.face_weight * face_conf +
                self.voice_weight * voice_conf
            )
            return {
                'identity': face_name,
                'person_id': face_result.get('person_id'),
                'confidence': min(fused_confidence * 1.1, 1.0),
                'modality': 'face+voice',
                'status': 'CONFIRMED',
                'details': {
                    'face': face_result,
                    'voice': voice_result
                }
            }
        else:
            # ── 4b: Conflict → use higher-weighted modality ──────────
            if face_conf * self.face_weight >= voice_conf * self.voice_weight:
                primary = face_result
                primary_modality = 'face'
            else:
                primary = voice_result
                primary_modality = 'voice'

            confidence_diff = abs(face_conf - voice_conf)

            return {
                'identity': primary['name'],
                'person_id': primary.get('person_id'),
                'confidence': primary['confidence'] * 0.7,  # Penalty for conflict
                'modality': f'{primary_modality}_primary_conflict',
                'status': 'CONFLICT',
                'details': {
                    'face': face_result,
                    'voice': voice_result,
                    'conflict_info': {
                        'face_says': face_name,
                        'voice_says': voice_name,
                        'confidence_diff': confidence_diff,
                        'primary_modality': primary_modality
                    }
                }
            }
