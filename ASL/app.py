import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import joblib
import os
import time

from hand_tracking.detector import HandDetector
from utils.sentence_builder import get_suggestions
from speech.tts import speak_text

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "asl_model.pkl")

# ---------------- LOAD MODEL ----------------
model = joblib.load(MODEL_PATH)

# ---------------- HAND DETECTOR ----------------
detector = HandDetector()

# ---------------- STATE ----------------
sentence = ""
current_char = ""
last_predicted_char = None
char_locked = False
last_capture_time = 0
CAPTURE_COOLDOWN = 1.2  # seconds

# ---------------- UI ----------------
root = tk.Tk()
root.title("Sign Language To Text Conversion")
root.geometry("1200x750")
root.configure(bg="#eeeeee")

# ---------------- TITLE ----------------
tk.Label(
    root,
    text="Sign Language To Text Conversion",
    font=("Courier", 28, "bold"),
    bg="#eeeeee"
).pack(pady=10)

# ---------------- MAIN FRAME ----------------
main_frame = tk.Frame(root, bg="#eeeeee")
main_frame.pack()

video_label = tk.Label(main_frame, bg="black", width=500, height=400)
video_label.grid(row=0, column=0, padx=20)

skeleton_label = tk.Label(main_frame, bg="white", width=400, height=400)
skeleton_label.grid(row=0, column=1, padx=20)

# ---------------- TEXT ----------------
text_frame = tk.Frame(root, bg="#eeeeee")
text_frame.pack(pady=15)

char_label = tk.Label(text_frame, text="Character : ", font=("Courier", 18), bg="#eeeeee")
char_label.pack(anchor="w")

sentence_label = tk.Label(
    text_frame,
    text="Sentence : ",
    font=("Courier", 18),
    bg="#eeeeee",
    wraplength=1000,
    justify="left"
)
sentence_label.pack(anchor="w", pady=5)

# ---------------- SUGGESTIONS ----------------
tk.Label(
    root,
    text="Suggestions",
    font=("Courier", 18, "bold"),
    fg="red",
    bg="#eeeeee"
).pack(pady=5)

suggestion_frame = tk.Frame(root, bg="#eeeeee")
suggestion_frame.pack(pady=5)

suggestion_buttons = []

def apply_suggestion(word):
    global sentence
    parts = sentence.split(" ")
    parts[-1] = word
    sentence = " ".join(parts)
    sentence_label.config(text=f"Sentence : {sentence}")

for _ in range(4):
    btn = tk.Button(
        suggestion_frame,
        text="",
        width=10,
        height=2,
        font=("Courier", 14)
    )
    btn.pack(side="left", padx=10)
    suggestion_buttons.append(btn)

# ---------------- ACTION BUTTONS ----------------
action_frame = tk.Frame(root, bg="#eeeeee")
action_frame.pack(pady=15)

def clear_text():
    global sentence
    sentence = ""
    sentence_label.config(text="Sentence : ")

def speak_sentence():
    speak_text(sentence)

tk.Button(action_frame, text="Clear", width=10, height=2, font=("Courier", 14), command=clear_text).pack(side="left", padx=5)
tk.Button(action_frame, text="Speak", width=10, height=2, font=("Courier", 14), command=speak_sentence).pack(side="left", padx=5)

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

# ---------------- HAND CONNECTIONS ----------------
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

# ---------------- UPDATE LOOP ----------------
def update_frame():
    global sentence, current_char, char_locked, last_capture_time

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame = cv2.flip(frame, 1)
    landmarks = detector.detect(frame)

    # CAMERA VIEW
    cam_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cam_img = ImageTk.PhotoImage(Image.fromarray(cam_rgb))
    video_label.configure(image=cam_img)
    video_label.image = cam_img

    # SKELETON CANVAS
    skel = np.ones((400, 400, 3), dtype=np.uint8) * 255

    if landmarks is not None and len(landmarks) == 63:
        X = np.array(landmarks).reshape(1, -1)
        pred = model.predict(X)[0]
        char = chr(pred + 65)
        current_char = char

        now = time.time()

        if not char_locked and now - last_capture_time > CAPTURE_COOLDOWN:
            sentence += char
            char_locked = True
            last_capture_time = now

        pts = np.array(landmarks).reshape(-1, 3)[:, :2]
        pts = pts * [400, 400]

        for i, j in HAND_CONNECTIONS:
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            cv2.line(skel, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        for x, y in pts:
            cv2.circle(skel, (int(x), int(y)), 4, (0, 255, 0), -1)

    else:
        char_locked = False  # unlock when hand removed

    skel_img = ImageTk.PhotoImage(Image.fromarray(skel))
    skeleton_label.configure(image=skel_img)
    skeleton_label.image = skel_img

    # UPDATE TEXT
    char_label.config(text=f"Character : {current_char}")
    sentence_label.config(text=f"Sentence : {sentence}")

    # UPDATE SUGGESTIONS
    suggestions = get_suggestions(sentence)
    for i, btn in enumerate(suggestion_buttons):
        if i < len(suggestions):
            btn.config(text=suggestions[i], command=lambda w=suggestions[i]: apply_suggestion(w))
        else:
            btn.config(text="", command=lambda: None)

    root.after(10, update_frame)

update_frame()
root.mainloop()
cap.release()