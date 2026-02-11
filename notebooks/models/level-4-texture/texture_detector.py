import os
import cv2
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

from texture_classifiers import MesoInception4

MODEL_PATH = "weights/MesoInception_DF.h5"
FRAME_SKIP = 5
IMG_SIZE = 256
THRESHOLD = 0.5     

# ==============================
# LOAD OFFICIAL MODEL
# ==============================
model = MesoInception4()
model.load(MODEL_PATH)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def analyze_texture(video_path, verbose=False):

    result = {
        "success": False,
        "video_path": video_path,
        "frames_analyzed": 0,
        "fake_ratio": 0,
        "avg_score": 0,
        "frame_scores": []
    }

    try:
        cap = cv2.VideoCapture(video_path)
        frame_id = 0
        scores = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % FRAME_SKIP == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    face = frame[y:y+h, x:x+w]
                    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                    face = face.astype("float32") / 255.0
                    face = np.expand_dims(face, axis=0)

                    pred = model.predict(face, verbose=0)[0][0]
                    scores.append(float(pred))

            frame_id += 1

        cap.release()

        if len(scores) == 0:
            result["reason"] = "no_faces_detected"
            return result

        scores = np.array(scores)
        avg_score = float(np.mean(scores))
        fake_ratio = float(np.mean(scores > THRESHOLD))

        result.update({
            "success": True,
            "frames_analyzed": len(scores),
            "fake_ratio": fake_ratio,
            "avg_score": avg_score,
            "frame_scores": scores.tolist()
        })

    except Exception as e:
        result["reason"] = str(e)

    return result
