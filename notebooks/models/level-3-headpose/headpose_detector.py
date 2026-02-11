import os
import cv2
import mediapipe as mp
import numpy as np


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


def analyze_headpose(
    video_path,
    verbose=False,
    save_debug=True,
    debug_dir="debug_headpose",
    max_debug_frames=100
):
    """
    Head pose analysis with MediaPipe + debug frame export
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"success": False, "reason": "cannot_open_video"}

    if save_debug:
        os.makedirs(debug_dir, exist_ok=True)

    pitch_vals = []
    yaw_vals = []
    roll_vals = []

    debug_frame_id = 0
    total_frames = 0

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                continue

            face = results.multi_face_landmarks[0]

            # ===== Landmark extraction =====
            nose = face.landmark[1]
            left_eye = face.landmark[33]
            right_eye = face.landmark[263]

            dx = right_eye.x - left_eye.x
            dy = right_eye.y - left_eye.y

            roll = np.arctan2(dy, dx)
            yaw = nose.x - 0.5
            pitch = nose.y - 0.5

            pitch_vals.append(pitch)
            yaw_vals.append(yaw)
            roll_vals.append(roll)

            # ===== Debug frame export =====
            if save_debug and debug_frame_id < max_debug_frames:

                h, w, _ = frame.shape

                # Draw landmarks
                annotated = frame.copy()
                mp_drawing.draw_landmarks(
                    annotated,
                    face,
                    mp_face_mesh.FACEMESH_CONTOURS
                )

                # Overlay pose text
                cv2.putText(
                    annotated,
                    f"Pitch: {pitch:.3f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    annotated,
                    f"Yaw: {yaw:.3f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    annotated,
                    f"Roll: {roll:.3f}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                debug_path = os.path.join(
                    debug_dir,
                    f"frame_{debug_frame_id:04d}.jpg"
                )

                cv2.imwrite(debug_path, annotated)
                debug_frame_id += 1

    cap.release()

    if len(pitch_vals) == 0:
        return {"success": False, "reason": "no_face_detected"}

    pitch = np.array(pitch_vals)
    yaw = np.array(yaw_vals)
    roll = np.array(roll_vals)

    # ===== Advanced motion metrics =====
    movements = np.stack([pitch, yaw, roll], axis=1)
    diff = np.diff(movements, axis=0)

    speed = np.linalg.norm(diff, axis=1)
    speed_variance = float(np.var(speed))
    avg_speed = float(np.mean(speed))

    pose_variance = float(
        np.var(pitch) +
        np.var(yaw) +
        np.var(roll)
    )

    # ===== Heuristic =====
    suspicious = False
    reasons = []
    print("Speed Variance:", speed_variance)
    if speed_variance < 1e-6:
        suspicious = True
        reasons.append("too_smooth_motion")

    if speed_variance > 0.01:
        suspicious = True
        reasons.append("jittery_motion")

    result = {
        "success": True,
        "frames_analyzed": len(pitch_vals),
        "total_video_frames": total_frames,
        "avg_pitch": float(np.mean(pitch)),
        "avg_yaw": float(np.mean(yaw)),
        "avg_roll": float(np.mean(roll)),
        "pose_variance": pose_variance,
        "avg_speed": avg_speed,
        "speed_variance": speed_variance,
        "debug_frames_saved": debug_frame_id,
        "suspicious": suspicious,
        "reasons": reasons
    }

    if verbose:
        print("\nHead Pose Summary:")
        print("Frames analyzed:", result["frames_analyzed"])
        print("Pose variance:", pose_variance)
        print("Speed variance:", speed_variance)
        print("Debug frames saved:", debug_frame_id)
        print("Suspicious" if suspicious else "Normal")

    return result
