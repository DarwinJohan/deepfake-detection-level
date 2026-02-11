import os
import json
from datetime import datetime

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

from emotion_detector import analyze_emotion


# ==============================
# ANALYZE SINGLE VIDEO
# ==============================
def analyze_video(video_path, verbose=False):
    """
    Analyze a single video using emotion detector only
    """

    result = {
        "video_path": video_path,
        "emotion": None,
        "prediction": None,
        "confidence": 0
    }

    try:
        emotion_result = analyze_emotion(video_path, verbose=verbose)

        if not emotion_result["success"]:
            result["error"] = emotion_result.get("reason", "emotion_failed")
            return result

        result["emotion"] = emotion_result

        # Simple prediction rule
        if emotion_result["suspicious"]:
            result["prediction"] = "FAKE"
            result["confidence"] = 0.85
        else:
            result["prediction"] = "REAL"
            result["confidence"] = 0.80

    except Exception as e:
        result["error"] = str(e)

    return result


# ==============================
# PRINT DETAILS
# ==============================
def print_video_details(filename, result, label):

    prediction = result.get("prediction", "ERROR")
    status = "OK" if prediction == label else "WRONG"

    print(f"\n{status} {filename}")
    print(f"   Label: {label}  |  Predicted: {prediction}")

    if not result.get("emotion"):
        print("\n   EMOTION ANALYSIS: ERROR")
        print("   " + "-" * 50)
        return

    emotion = result["emotion"]

    print("\n   EMOTION ANALYSIS:")
    print(f"      Status: {'SUSPICIOUS' if emotion['suspicious'] else 'NORMAL'}")

    if emotion["suspicious"]:
        print(f"      Reasons: {', '.join(emotion['reasons'])}")

    print("      Emotions detected:")

    for emo, count in emotion["emotion_frequency"].items():
        pct = (count / emotion["total_faces"]) * 100
        print(f"         - {emo}: {count} ({pct:.1f}%)")

    print(f"      Dominant: {emotion['dominant_emotion']}")
    print(f"      Avg Confidence: {emotion['avg_confidence']:.1f}%")
    print(f"      Diversity: {emotion['emotion_diversity']}")

    print("   " + "-" * 50)


# ==============================
# DATASET PROCESSING
# ==============================
def process_dataset(fake_dir="fake", real_dir="real", output_file="emotion_results.json"):

    print("\n" + "=" * 60)
    print("EMOTION DATASET PROCESSING")
    print("=" * 60)

    video_ext = [".mp4", ".avi", ".mov", ".mkv", ".flv"]

    def collect_videos(folder):
        if not os.path.exists(folder):
            return []
        return [
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if any(f.lower().endswith(ext) for ext in video_ext)
        ]

    fake_videos = collect_videos(fake_dir)
    real_videos = collect_videos(real_dir)

    print(f"Found {len(fake_videos)} fake videos")
    print(f"Found {len(real_videos)} real videos")

    total = len(fake_videos) + len(real_videos)

    if total == 0:
        print("No videos found.")
        return

    all_results = []

    # ======================
    # FAKE BATCH
    # ======================
    print("\n" + "=" * 60)
    print("Batch: FAKE")
    print("=" * 60)

    for vid in fake_videos:
        name = os.path.basename(vid)
        print(f"\nProcessing: {name}...")
        r = analyze_video(vid)
        r["ground_truth"] = "FAKE"
        all_results.append(r)
        print_video_details(name, r, "FAKE")

    # ======================
    # REAL BATCH
    # ======================
    print("\n" + "=" * 60)
    print("Batch: REAL")
    print("=" * 60)

    for vid in real_videos:
        name = os.path.basename(vid)
        print(f"\nProcessing: {name}...")
        r = analyze_video(vid)
        r["ground_truth"] = "REAL"
        all_results.append(r)
        print_video_details(name, r, "REAL")

    # ======================
    # METRICS
    # ======================
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    tp = tn = fp = fn = correct = 0

    for r in all_results:
        pred = r.get("prediction")
        gt = r["ground_truth"]

        if not pred:
            continue

        if pred == gt:
            correct += 1

        if gt == "FAKE" and pred == "FAKE":
            tp += 1
        elif gt == "REAL" and pred == "REAL":
            tn += 1
        elif gt == "REAL" and pred == "FAKE":
            fp += 1
        elif gt == "FAKE" and pred == "REAL":
            fn += 1

    total_valid = tp + tn + fp + fn

    acc = (correct / total_valid * 100) if total_valid else 0
    precision = (tp / (tp + fp) * 100) if (tp + fp) else 0
    recall = (tp / (tp + fn) * 100) if (tp + fn) else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    print(f"\nAccuracy: {acc:.1f}% ({correct}/{total_valid})")

    print("\nMetrics:")
    print(f"   Precision: {precision:.1f}%")
    print(f"   Recall:    {recall:.1f}%")
    print(f"   F1-score:  {f1:.1f}%")

    print("\nConfusion Matrix:")
    print(f"   TP: {tp}")
    print(f"   TN: {tn}")
    print(f"   FP: {fp}")
    print(f"   FN: {fn}")

    # ======================
    # SAVE JSON
    # ======================
    output = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": {
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn
            }
        },
        "results": all_results
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to: {output_file}")
    print("=" * 60)

    return all_results


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    process_dataset(
        fake_dir="fake",
        real_dir="real",
        output_file="emotion_evaluation.json"
    )
