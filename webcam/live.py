import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# ---------- 1. Model: NumPy forward pass (same as Colab) ----------
w = np.load("asl_mlp_numpy.npz")
LETTERS = [str(l) for l in w["letters"]]


def forward(x):
    h = np.maximum(0, x @ w["net.0.weight"].T + w["net.0.bias"])
    h = np.maximum(0, h @ w["net.2.weight"].T + w["net.2.bias"])
    return h @ w["net.4.weight"].T + w["net.4.bias"]


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def to_features(landmarks, flip=False):
    pts = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
    if flip:
        pts[:, 0] = 1 - pts[:, 0]
    pts = pts - pts[0]
    pts = pts / np.abs(pts).max()
    return pts.flatten()


# ---------- 2. MediaPipe in VIDEO mode ----------
detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.3,
))

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
                    (18, 19), (19, 20), (0, 17)]

# ---------- 3. Webcam loop ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Can't open the webcam. Allow camera access for Terminal in System Settings.")

recent = deque(maxlen=10)
flip = False
last_ts = 0
fps, frames, t0 = 0.0, 0, time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ts = max(int(time.time() * 1000), last_ts + 1)
    last_ts = ts
    result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

    display = cv2.flip(frame, 1)
    h, wd = display.shape[:2]
    text, color, top3_text = "No hand", (0, 0, 255), ""

    if result.hand_landmarks:
        lms = result.hand_landmarks[0]
        probs = softmax(forward(to_features(lms, flip)))
        recent.append(probs)
        avg = np.mean(recent, axis=0)
        k = int(avg.argmax())
        if avg[k] > 0.6:
            text, color = f"{LETTERS[k]}  {avg[k]:.0%}", (0, 200, 0)
        else:
            text, color = f"{LETTERS[k]}?  {avg[k]:.0%}", (0, 200, 255)
        top3_text = "   ".join(f"{LETTERS[i]} {avg[i]:.0%}" for i in avg.argsort()[::-1][:3])

        pts = [(int((1 - lm.x) * wd), int(lm.y * h)) for lm in lms]
        for a, b in HAND_CONNECTIONS:
            cv2.line(display, pts[a], pts[b], (255, 0, 0), 2)
        for p in pts:
            cv2.circle(display, p, 4, (0, 0, 255), -1)
    else:
        recent.clear()

    frames += 1
    if time.time() - t0 >= 1:
        fps, frames, t0 = frames / (time.time() - t0), 0, time.time()

    cv2.putText(display, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 5)
    cv2.putText(display, top3_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(display, f"FPS {fps:.0f}   flip {'ON' if flip else 'OFF'}   [f] flip   [q] quit",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("ASL live", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("f"):
        flip = not flip
        recent.clear()

cap.release()
cv2.destroyAllWindows()
