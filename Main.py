import os
from datetime import datetime

import cv2
import face_recognition
import numpy as np

IMAGES_DIR = "images"
ATTENDANCE_FILE = "attendance.csv"


def load_known_faces():
    encodings, names = [], []
    for file in os.listdir(IMAGES_DIR):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        image = face_recognition.load_image_file(os.path.join(IMAGES_DIR, file))
        found = face_recognition.face_encodings(image)
        if found:
            encodings.append(found[0])
            names.append(os.path.splitext(file)[0])
    return encodings, names


def already_marked(name, today):
    if not os.path.exists(ATTENDANCE_FILE):
        return False
    with open(ATTENDANCE_FILE) as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(",")
            if len(parts) >= 2 and parts[0] == name and parts[1] == today:
                return True
    return False


def mark_attendance(name):
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    if already_marked(name, today):
        return
    new_file = not os.path.exists(ATTENDANCE_FILE)
    with open(ATTENDANCE_FILE, "a") as f:
        if new_file:
            f.write("Name,Date,Time\n")
        f.write(f"{name},{today},{now.strftime('%H:%M:%S')}\n")
    print(f"Marked present: {name}")


def main():
    known_encodings, known_names = load_known_faces()
    print(f"Loaded {len(known_names)} known faces")
    cap = cv2.VideoCapture(0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)

        for enc, (top, right, bottom, left) in zip(encodings, locations):
            name = "Unknown"
            if known_encodings:
                distances = face_recognition.face_distance(known_encodings, enc)
                best = int(np.argmin(distances))
                if distances[best] < 0.5:
                    name = known_names[best]
                    mark_attendance(name)
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
