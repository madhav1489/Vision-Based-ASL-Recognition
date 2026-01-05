import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(""))

from hand_tracking.detector import HandDetector

DATA_DIR = "data_processed"
os.makedirs(DATA_DIR, exist_ok=True)

X_path = os.path.join(DATA_DIR, "X.npy")
y_path = os.path.join(DATA_DIR, "y.npy")

cap = cv2.VideoCapture(0)
detector = HandDetector()

label = int(input("Enter label (A=0 ... Z=25): "))

X, y = [], []

print("Press 'q' to stop")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    landmarks = detector.detect(frame)

    if landmarks is not None:
        X.append(landmarks)
        y.append(label)

        cv2.putText(
            frame,
            "CAPTURED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

    cv2.imshow("Capture Data", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

X = np.array(X)
y = np.array(y)

if X.shape[0] == 0:
    print("❌ No data captured")
    exit()

if os.path.exists(X_path):
    X_old = np.load(X_path)
    y_old = np.load(y_path)
    X = np.vstack((X_old, X))
    y = np.concatenate((y_old, y))

np.save(X_path, X)
np.save(y_path, y)

print("✅ Data saved:", X.shape)
