"""
Identity Database Module.
Stores and manages face/speaker embeddings and metadata.
Privacy-aware: only stores computed embeddings, never raw images or audio.
"""

import json
import os
import numpy as np
from datetime import datetime


class IdentityDatabase:
    """JSON-backed identity database storing embeddings and metadata."""

    def __init__(self, db_path=None):
        # Lazy import to avoid circular dependency
        if db_path is None:
            from utils.config import DB_PATH
            db_path = db_path or DB_PATH
        self.db_path = db_path
        self.identities = {}
        self.load()

    def load(self):
        """Load identity database from disk."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                self.identities = {}
                for pid, info in data.items():
                    self.identities[pid] = {
                        'name': info['name'],
                        'face_embeddings': [np.array(e) for e in info.get('face_embeddings', [])],
                        'speaker_embeddings': [np.array(e) for e in info.get('speaker_embeddings', [])],
                        'metadata': info.get('metadata', {}),
                        'created_at': info.get('created_at', ''),
                        'updated_at': info.get('updated_at', '')
                    }
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠️ Warning: Could not load database: {e}")
                self.identities = {}
        else:
            self.identities = {}

    def save(self):
        """Persist identity database to disk."""
        data = {}
        for pid, info in self.identities.items():
            data[pid] = {
                'name': info['name'],
                'face_embeddings': [e.tolist() for e in info.get('face_embeddings', [])],
                'speaker_embeddings': [e.tolist() for e in info.get('speaker_embeddings', [])],
                'metadata': info.get('metadata', {}),
                'created_at': info.get('created_at', ''),
                'updated_at': info.get('updated_at', '')
            }
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2)

    def add_identity(self, name, face_embedding=None, speaker_embedding=None, metadata=None):
        """Add a new identity to the database. Returns the generated person ID."""
        # Generate a unique person ID
        existing_nums = []
        for pid in self.identities:
            try:
                existing_nums.append(int(pid.split('_')[1]))
            except (IndexError, ValueError):
                pass
        next_num = max(existing_nums, default=0) + 1
        pid = f"person_{next_num:03d}"

        now = datetime.now().isoformat()
        self.identities[pid] = {
            'name': name,
            'face_embeddings': [face_embedding] if face_embedding is not None else [],
            'speaker_embeddings': [speaker_embedding] if speaker_embedding is not None else [],
            'metadata': metadata or {},
            'created_at': now,
            'updated_at': now
        }
        self.save()
        return pid

    def add_face_embedding(self, pid, embedding):
        """Add an additional face embedding sample for an existing identity."""
        if pid in self.identities:
            self.identities[pid]['face_embeddings'].append(embedding)
            self.identities[pid]['updated_at'] = datetime.now().isoformat()
            self.save()
        else:
            raise KeyError(f"Identity {pid} not found in database")

    def add_speaker_embedding(self, pid, embedding):
        """Add an additional speaker embedding sample for an existing identity."""
        if pid in self.identities:
            self.identities[pid]['speaker_embeddings'].append(embedding)
            self.identities[pid]['updated_at'] = datetime.now().isoformat()
            self.save()
        else:
            raise KeyError(f"Identity {pid} not found in database")

    def get_all_face_embeddings(self):
        """Get all stored face embeddings as (person_id, name, embedding) tuples."""
        result = []
        for pid, info in self.identities.items():
            for emb in info['face_embeddings']:
                result.append((pid, info['name'], emb))
        return result

    def get_all_speaker_embeddings(self):
        """Get all stored speaker embeddings as (person_id, name, embedding) tuples."""
        result = []
        for pid, info in self.identities.items():
            for emb in info['speaker_embeddings']:
                result.append((pid, info['name'], emb))
        return result

    def get_identity(self, pid):
        """Get identity info by person_id."""
        return self.identities.get(pid)

    def get_all_identities(self):
        """Get all identities."""
        return self.identities

    def remove_identity(self, pid):
        """Remove an identity from the database."""
        if pid in self.identities:
            del self.identities[pid]
            self.save()

    def get_identity_count(self):
        """Get the total number of registered identities."""
        return len(self.identities)

    def clear(self):
        """Clear all identities from the database."""
        self.identities = {}
        self.save()
