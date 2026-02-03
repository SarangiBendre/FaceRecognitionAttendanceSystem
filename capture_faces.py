import cv2
import os
import sys

student_id = sys.argv[1]

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

dataset_path = f"dataset/{student_id}"
os.makedirs(dataset_path, exist_ok=True)

cam = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face = gray[y:y+h, x:x+w]
        cv2.imwrite(f"{dataset_path}/{count}.jpg", face)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Register Face", frame)

    if cv2.waitKey(1) == 27 or count >= 20:
        break

cam.release()
cv2.destroyAllWindows()
