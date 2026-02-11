import os
import json
from datetime import datetime

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

from texture_detector import analyze_texture


# ==============================
# ANALYZE SINGLE VIDEO
# ==============================
def analyze_video(video_path, verbose=False):

    result = {
        "video_path": video_path,
        "texture": None,
        "prediction": None,
        "confidence": 0
    }

    try:
        tex = analyze_texture(video_path, verbose=verbose)

        if not tex["success"]:
            result["error"] = tex.get("reason", "texture_failed")
            return result

        result["texture"] = tex

        # Decision rule
        if tex["avg_score"] < 0.8:
            result["prediction"] = "FAKE"
            result["confidence"] = tex["avg_score"]
        else:
            result["prediction"] = "REAL"
            result["confidence"] = 1 - tex["avg_score"]

    except Exception as e:
        result["error"] = str(e)

    return result


# ==============================
# PRINT DETAILS
# ==============================
def print_video_details(filename, result, label):

    prediction = result.get("prediction", "ERROR")
    status = "✅" if prediction == label else "❌"

    print(f"\n{status} {filename}")
    print(f"   Label: {label}  |  Predicted: {prediction}")

    if not result.get("texture"):
        print("   Texture analysis: ERROR")
        print(f"   Reason: {result.get('reason')}")

        return

    tex = result["texture"]

    print("   📊 TEXTURE ANALYSIS:")
    print(f"      Frames: {tex['frames_analyzed']}")
    print(f"      Avg score: {tex['avg_score']:.4f}")
    print(f"      Fake ratio: {tex['fake_ratio']:.2f}")


# ==============================
# DATASET PROCESSING
# ==============================
def process_dataset(fake_dir="fake", real_dir="real", output_file="texture_results.json"):

    print("\n" + "=" * 60)
    print("🧪 TEXTURE DATASET PROCESSING")
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

    print(f"📁 Found {len(fake_videos)} fake videos")
    print(f"📁 Found {len(real_videos)} real videos")

    all_results = []

    # FAKE
    print("\n" + "=" * 60)
    print("Batch: FAKE")
    print("=" * 60)

    for vid in fake_videos:
        name = os.path.basename(vid)
        print(f"\n🎬 Processing: {name}...")
        r = analyze_video(vid)
        r["ground_truth"] = "FAKE"
        all_results.append(r)
        print_video_details(name, r, "FAKE")

    # REAL
    print("\n" + "=" * 60)
    print("Batch: REAL")
    print("=" * 60)

    for vid in real_videos:
        name = os.path.basename(vid)
        print(f"\n🎬 Processing: {name}...")
        r = analyze_video(vid)
        r["ground_truth"] = "REAL"
        all_results.append(r)
        print_video_details(name, r, "REAL")

    # ======================
    # METRICS
    # ======================
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

    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)

    print(f"🎯 Accuracy: {acc:.1f}% ({correct}/{total_valid})")

    print("\n📈 Metrics:")
    print(f"   Precision: {precision:.1f}%")
    print(f"   Recall:    {recall:.1f}%")
    print(f"   F1-score:  {f1:.1f}%")

    print("\n🔍 Confusion Matrix:")
    print(f"   TP: {tp}")
    print(f"   TN: {tn}")
    print(f"   FP: {fp}")
    print(f"   FN: {fn}")

    # SAVE JSON
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

    print(f"\n💾 Saved to: {output_file}")
    print("=" * 60)

    return all_results


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    process_dataset(
        fake_dir="fake",
        real_dir="real",
        output_file="texture_evaluation.json"
    )
