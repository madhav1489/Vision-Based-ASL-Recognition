import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mpDraw = mp.solutions.drawing_utils

    def detect(self, frame):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        landmarks = []

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(
                    frame,
                    handLms,
                    self.mpHands.HAND_CONNECTIONS,
                    self.mpDraw.DrawingSpec(color=(0,255,0), thickness=2),
                    self.mpDraw.DrawingSpec(color=(0,255,0), thickness=2)
                )

                for lm in handLms.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

        if len(landmarks) == 63:
            return landmarks

        return None
