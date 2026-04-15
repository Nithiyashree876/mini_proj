"""
Face Recognition Module.
Extracts face embeddings using a combination of pixel features, LBP histograms,
and HOG-like features, then matches against stored identities via cosine similarity.
"""

import cv2
import numpy as np
from utils.config import FACE_EMBEDDING_SIZE, FACE_RECOGNITION_THRESHOLD


class FaceRecognizer:
    """Face recognition via multi-feature embedding extraction and cosine matching."""

    def __init__(self, database):
        import os
        self.database = database
        self.threshold = 0.363  # Standard cosine similarity threshold for SFace (Cosine)
        
        # Load pre-trained SFace DL model
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'sface.onnx'))
        try:
            self.recognizer = cv2.FaceRecognizerSF.create(model=model_path, config="")
        except Exception as e:
            raise RuntimeError(f"Failed to load SFace model: {e}. Ensure models from opencv zoo are downloaded.")

    def extract_embedding(self, original_frame, yunet_face):
        """Extract a 128D deep feature embedding from a face image using SFace.

        Args:
            original_frame: The full BGR video frame
            yunet_face: The 15-float array face detection result from YuNet

        Returns:
            L2-normalized feature vector (numpy array)
        """
        # Align the face structurally based on YuNet landmarks (eyes, nose, mouth)
        aligned_face = self.recognizer.alignCrop(original_frame, yunet_face)
        
        # Extract 128D deep learning feature
        feature = self.recognizer.feature(aligned_face)
        
        # Features come out as 1x128 array -> flatten to 1D
        embedding = feature.flatten()
        return embedding

    def recognize(self, original_frame, yunet_face):
        """Recognize a face by matching its embedding against the database.

        Args:
            original_frame: The full BGR video frame
            yunet_face: 15-float array face detection result from YuNet

        Returns:
            Dict with keys: name, person_id, confidence, embedding
        """
        embedding = self.extract_embedding(original_frame, yunet_face)

        known_faces = self.database.get_all_face_embeddings()
        if not known_faces:
            return {
                'name': 'Unknown',
                'person_id': None,
                'confidence': 0.0,
                'embedding': embedding
            }

        best_match = None
        best_score = -1.0

        for pid, name, stored_emb in known_faces:
            # Skip if embedding dimensions don't match (version mismatch)
            if len(embedding) != len(stored_emb):
                continue

            # Cosine similarity (both vectors are already L2-normalized)
            score = float(np.dot(embedding, stored_emb))

            if score > best_score:
                best_score = score
                best_match = (pid, name)

        if best_match and best_score >= self.threshold:
            return {
                'name': best_match[1],
                'person_id': best_match[0],
                # Map SFace score [threshold, 1.0] to a percentage [0.5, 1.0] for UI
                'confidence': 0.5 + 0.5 * ((best_score - self.threshold) / (1.0 - self.threshold)),
                'embedding': embedding
            }
        else:
            return {
                'name': 'Unknown',
                'person_id': None,
                'confidence': max(0.0, best_score),
                'embedding': embedding
            }
