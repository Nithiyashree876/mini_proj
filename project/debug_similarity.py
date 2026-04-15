import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vision.face_recognizer import FaceRecognizer
from utils.database import IdentityDatabase

def run():
    rec = FaceRecognizer(None)
    
    # Generate random face 1
    np.random.seed(42)
    face1 = np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8)
    
    # face2 is face1 but with slightly different noise
    face2 = np.clip(face1.astype(np.int16) + np.random.randint(-15, 15, face1.shape), 0, 255).astype(np.uint8)
    
    # completely different face
    np.random.seed(99)
    face3 = np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8)

    emb1 = rec.extract_embedding(face1)
    emb2 = rec.extract_embedding(face2)
    emb3 = rec.extract_embedding(face3)
    
    sim_same = float(np.dot(emb1, emb2))
    sim_diff = float(np.dot(emb1, emb3))
    
    print(f"Similarity (Same Person, lit diff): {sim_same:.4f}")
    print(f"Similarity (Diff Person): {sim_diff:.4f}")
    
    print(f"Embedding length: {len(emb1)}")
    
if __name__ == '__main__':
    run()
