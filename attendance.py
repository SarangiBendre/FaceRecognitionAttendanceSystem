import cv2
import pandas as pd
from datetime import datetime
import os
import sys

# Load students
students = {}
df = pd.read_csv("students.csv")
for _, row in df.iterrows():
    students[int(row["ID"])] = row["Name"]

# Load model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trained_model.yml")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cam = cv2.VideoCapture(0)

# Attendance file
if not os.path.exists("attendance"):
    os.mkdir("attendance")

att_file = "attendance/attendance.csv"
if not os.path.exists(att_file):
    pd.DataFrame(columns=["ID","Name","Date","Time"]).to_csv(att_file, index=False)

success_output = "FAILED"

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face)

        if confidence < 80:
            name = students.get(label, "Unknown")
            now = datetime.now()

            df = pd.read_csv(att_file)
            df.loc[len(df)] = [
                label,
                name,
                now.strftime("%d-%m-%Y"),
                now.strftime("%H:%M:%S")
            ]
            df.to_csv(att_file, index=False)

            success_output = f"SUCCESS|{label}|{name}|{now.strftime('%H:%M:%S')}"

            cam.release()
            cv2.destroyAllWindows()

            # 🔴 THIS IS THE ONLY THING FLASK NEEDS
            print(success_output)
            sys.exit(0)

    cv2.imshow("Mark Attendance", frame)
    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()

print(success_output)
sys.exit(0)
