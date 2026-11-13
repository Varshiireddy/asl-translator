import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

LETTERS = [str(l) for l in np.load("asl_mlp_numpy.npz")["letters"]]
PER_LETTER = 15   # photos per letter
EVERY = 4         # while capturing, keep every 4th frame (so the 15 aren't identical)

detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.3,
))

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
                    (18, 19), (19, 20), (0, 17)]


def crop_hand(rgb, landmarks, size=64):
    h, w = rgb.shape[:2]
    xs = np.array([lm.x for lm in landmarks]) * w
    ys = np.array([lm.y for lm in landmarks]) * h
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) * 1.3
    pad = int(side) + 1
    padded = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    x0, y0 = int(cx - side / 2) + pad, int(cy - side / 2) + pad
    s = int(side)
    return cv2.resize(padded[y0:y0 + s, x0:x0 + s], (size, size))


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Can't open the webcam. Allow camera access for Terminal in System Settings.")

lms_out, crops, labels = [], [], []
i, got, capturing, hand_frames, last_ts = 0, 0, False, 0, 0

while i < len(LETTERS):
    ok, frame = cap.read()
    if not ok:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ts = max(int(time.time() * 1000), last_ts + 1)
    last_ts = ts
    res = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

    display = frame.copy()
    h, w = display.shape[:2]

    if res.hand_landmarks:
        lms = res.hand_landmarks[0]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
        for a, b in HAND_CONNECTIONS:
            cv2.line(display, pts[a], pts[b], (255, 0, 0), 2)
        hand_frames += 1
        if capturing and hand_frames % EVERY == 0:
            lms_out.append([[lm.x, lm.y, lm.z] for lm in lms])
            crops.append(crop_hand(rgb, lms))
            labels.append(i)
            got += 1
            if got >= PER_LETTER:
                capturing, got = False, 0
                i += 1
                if i >= len(LETTERS):
                    break

    cv2.putText(display, f"Show: {LETTERS[i]}   ({i + 1}/{len(LETTERS)})", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
    status = f"Capturing {got}/{PER_LETTER} - move your hand a little" if capturing \
        else "SPACE = capture    S = skip letter    Q = save and quit"
    cv2.putText(display, status, (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if capturing:
        cv2.rectangle(display, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)
    if not res.hand_landmarks:
        cv2.putText(display, "No hand", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.imshow("Collect my hands", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(" ") and not capturing:
        capturing, got = True, 0
    elif key == ord("s") and not capturing:
        i += 1
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

np.savez_compressed("my_hands.npz", lms=np.array(lms_out, dtype=np.float32),
                    crops=np.array(crops, dtype=np.uint8), y=np.array(labels))
print(f"Saved {len(labels)} photos for {len(set(labels))} letters -> my_hands.npz")
