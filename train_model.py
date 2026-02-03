import cv2
import os
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
faces = []
labels = []

for folder in os.listdir("dataset"):
    for img in os.listdir(f"dataset/{folder}"):
        path = f"dataset/{folder}/{img}"
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        faces.append(image)
        labels.append(int(folder))

recognizer.train(faces, np.array(labels))
recognizer.save("trained_model.yml")

print("Training completed")
