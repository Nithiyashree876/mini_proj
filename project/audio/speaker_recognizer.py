"""
Speaker Recognition Module.
Extracts speaker embeddings from audio using MFCC statistics
(mean, std, min, max of MFCCs and their delta/delta-delta),
then matches against stored speaker profiles via cosine similarity.
"""

import numpy as np


class SpeakerRecognizer:
    """MFCC-based speaker recognition with cosine similarity matching."""

    def __init__(self, database):
        from utils.config import AUDIO_SAMPLE_RATE, AUDIO_N_MFCC, SPEAKER_RECOGNITION_THRESHOLD
        self.database = database
        self.sample_rate = AUDIO_SAMPLE_RATE
        self.n_mfcc = AUDIO_N_MFCC
        self.threshold = SPEAKER_RECOGNITION_THRESHOLD
        self._librosa = None

    def _get_librosa(self):
        """Lazy-load librosa to avoid slow startup when not needed."""
        if self._librosa is None:
            try:
                import librosa
                self._librosa = librosa
            except ImportError:
                raise ImportError(
                    "librosa is required for speaker recognition. "
                    "Install it with: pip install librosa"
                )
        return self._librosa

    def extract_embedding(self, audio_data, sr=None):
        """Extract a speaker embedding from raw audio data.

        Computes MFCCs + delta + delta-delta, then summarizes each
        coefficient with statistical measures (mean, std, min, max)
        to create a fixed-length speaker-characteristic vector.

        Args:
            audio_data: 1D numpy array of audio samples
            sr: Sample rate (uses default if None)

        Returns:
            L2-normalized speaker embedding vector
        """
        librosa = self._get_librosa()
        sr = sr or self.sample_rate

        # Ensure float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Extract MFCC coefficients
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=self.n_mfcc)

        # Compute delta and delta-delta (velocity and acceleration of features)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        # Compute summary statistics for each feature matrix
        features = []
        for feat_matrix in [mfccs, delta_mfccs, delta2_mfccs]:
            features.extend([
                np.mean(feat_matrix, axis=1),
                np.std(feat_matrix, axis=1),
                np.min(feat_matrix, axis=1),
                np.max(feat_matrix, axis=1),
            ])

        embedding = np.concatenate(features)

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def recognize(self, audio_data, sr=None):
        """Recognize a speaker from audio data.

        Args:
            audio_data: 1D numpy array of audio samples
            sr: Sample rate

        Returns:
            Dict with: name, person_id, confidence, embedding
        """
        embedding = self.extract_embedding(audio_data, sr)

        known_speakers = self.database.get_all_speaker_embeddings()
        if not known_speakers:
            return {
                'name': 'Unknown',
                'person_id': None,
                'confidence': 0.0,
                'embedding': embedding
            }

        best_match = None
        best_score = -1.0

        for pid, name, stored_emb in known_speakers:
            # Skip dimension mismatches
            if len(embedding) != len(stored_emb):
                continue

            # Cosine similarity
            score = float(np.dot(embedding, stored_emb))

            if score > best_score:
                best_score = score
                best_match = (pid, name)

        if best_match and best_score >= self.threshold:
            return {
                'name': best_match[1],
                'person_id': best_match[0],
                'confidence': best_score,
                'embedding': embedding
            }
        else:
            return {
                'name': 'Unknown',
                'person_id': None,
                'confidence': best_score if best_score > 0 else 0.0,
                'embedding': embedding
            }
