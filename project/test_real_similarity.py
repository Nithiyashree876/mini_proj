import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vision.face_recognizer import FaceRecognizer
from vision.face_detector import FaceDetector

def load_and_detect(img_path, detector):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read {img_path}")
        return None
    detections = detector.detect_faces(img)
    if not detections:
        print(f"No faces in {img_path}")
        return None
    return detections[0]['face_region']

def run():
    rec = FaceRecognizer(None)
    det = FaceDetector()
    
    faces = {}
    base_dir = r"c:\Users\LENOVO\OneDrive\Desktop\Face detection"
    for file in ["Dhanya.jpeg", "Lizania.jpeg", "Nithiya.jpeg"]:
        path = os.path.join(base_dir, file)
        face_img = load_and_detect(path, det)
        if face_img is not None:
            faces[file] = rec.extract_embedding(face_img)
            
    if len(faces) >= 2:
        keys = list(faces.keys())
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                k1, k2 = keys[i], keys[j]
                sim = float(np.dot(faces[k1], faces[k2]))
                print(f"Similarity b/w {k1} and {k2}: {sim:.4f}")

if __name__ == '__main__':
    run()
