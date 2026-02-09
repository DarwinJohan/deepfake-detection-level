import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque

# ------------------------
# Config
# ------------------------
DEBUG_DIR = "debug_frames"
os.makedirs(DEBUG_DIR, exist_ok=True)

EAR_THRESHOLD = 0.25  # threshold for blink
CONSEC_FRAMES = 2     # consecutive frames for blink

# Mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # iris detection
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark indices (from Mediapipe)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]

# ------------------------
# Helper functions
# ------------------------
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(landmarks, eye_idx, image_shape):
    h, w = image_shape
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_idx]
    A = euclidean(pts[1], pts[5])
    B = euclidean(pts[2], pts[4])
    C = euclidean(pts[0], pts[3])
    ear = (A + B) / (2.0 * C)
    return ear

# ------------------------
# Blink detection main
# ------------------------
def analyze_blink(video_path, max_frames=100, frame_interval=3, verbose=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"success": False, "reason": "cannot_open_video"}

    blink_count = 0
    counter = 0
    frame_count = 0
    processed_frames = 0
    ear_history = deque(maxlen=5)  # smoothing EAR
    debug_frames_saved = 0

    while cap.isOpened() and processed_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            processed_frames += 1
            continue

        landmarks = results.multi_face_landmarks[0].landmark
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_IDX, frame.shape[:2])
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, frame.shape[:2])
        ear = (left_ear + right_ear) / 2.0
        ear_history.append(ear)

        # Blink detection
        if ear < EAR_THRESHOLD:
            counter += 1
        else:
            if counter >= CONSEC_FRAMES:
                blink_count += 1
                if verbose:
                    print(f"   👁️ Blink #{blink_count} at frame {frame_count}")
            counter = 0

        # Draw eyes for debug
        for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
            x = int(landmarks[idx].x * frame.shape[1])
            y = int(landmarks[idx].y * frame.shape[0])
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Save debug frame
        debug_filename = os.path.join(DEBUG_DIR, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(debug_filename, frame)
        debug_frames_saved += 1

        processed_frames += 1

    cap.release()

    if not ear_history:
        return {"success": False, "reason": "no_faces_detected"}

    # Metrics
    avg_ear = float(np.mean(ear_history))
    std_ear = float(np.std(ear_history))
    fps = 30  # assume 30 fps
    duration_seconds = (processed_frames * frame_interval) / fps
    blink_rate_per_minute = (blink_count / duration_seconds) * 60 if duration_seconds > 0 else 0

# ------------------------
# Suspicious detection (tune for less false positive)
# ------------------------
    suspicious = False
    reasons = []

    # turunkan sensitivitas: rentang blink rate lebih lebar
    if blink_rate_per_minute < 5:  # sebelumnya 8
        suspicious = True
        reasons.append("low_blink_rate")
    elif blink_rate_per_minute > 50:  # sebelumnya 35
        suspicious = True
        reasons.append("high_blink_rate")

    # Variansi EAR rendah sekarang diabaikan sampai < 0.008
    if std_ear < 0.008:
        suspicious = True
        reasons.append("low_ear_variance")

    # Avg EAR ekstrem
    if avg_ear < 0.12 or avg_ear > 0.38:  # sedikit diperluas
        suspicious = True
        reasons.append("abnormal_ear")


    result = {
        "success": True,
        "blink_count": blink_count,
        "blink_rate_per_minute": blink_rate_per_minute,
        "avg_ear": avg_ear,
        "std_ear": std_ear,
        "processed_frames": processed_frames,
        "debug_frames_saved": debug_frames_saved,
        "suspicious": suspicious,
        "reasons": reasons
    }

    if verbose:
        print("\n👁️ Blink Detection Summary:")
        print(f"   Total blinks: {blink_count}")
        print(f"   Blink rate (/min): {blink_rate_per_minute:.1f}")
        print(f"   Avg EAR: {avg_ear:.3f}, Std EAR: {std_ear:.3f}")
        if suspicious:
            print(f"   ⚠️ Suspicious: {', '.join(reasons)}")
        else:
            print("   ✅ Looks normal")

    return result
