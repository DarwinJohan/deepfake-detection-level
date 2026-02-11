import os
import json
from datetime import datetime

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

from headpose_detector import analyze_headpose  # pastikan file ini ada


def analyze_video(video_path, verbose=False):
    """
    Analyze single video and return head pose results
    """
    results = {
        "video_path": video_path,
        "headpose": None,
        "prediction": None,
        "confidence": 0
    }

    try:
        # Head pose analysis
        hp_result = analyze_headpose(video_path, verbose=True)
        print("DEBUG:", hp_result)

        if hp_result["success"]:
            results["headpose"] = hp_result

            # Simple prediction rule
            if hp_result["suspicious"]:
                results["prediction"] = "FAKE"
                results["confidence"] = 0.8
            else:
                results["prediction"] = "REAL"
                results["confidence"] = 0.9

    except Exception as e:
        import traceback
        print("\nREAL ERROR:")
        traceback.print_exc()
        results["error"] = str(e)

    return results


def print_video_details(filename, result, label):
    """
    Print head pose analysis results
    """
    prediction = result.get("prediction", "ERROR")
    status = "OK" if prediction == label else "WRONG"

    print(f"\n{status} {filename}")
    print(f"   Label: {label}  |  Predicted: {prediction}")

    if result.get("headpose"):
        hp = result["headpose"]

        print(f"\n   HEAD POSE ANALYSIS:")
        print(f"      Status: {'SUSPICIOUS' if hp['suspicious'] else 'NORMAL'}")

        if hp["suspicious"]:
            print(f"      Reasons: {', '.join(hp['reasons'])}")

        # Optional metrics (adjust sesuai output detector kamu)
        if "avg_pitch" in hp:
            print(f"      Avg Pitch: {hp['avg_pitch']:.2f}")

        if "avg_yaw" in hp:
            print(f"      Avg Yaw: {hp['avg_yaw']:.2f}")

        if "avg_roll" in hp:
            print(f"      Avg Roll: {hp['avg_roll']:.2f}")

        if "pose_variance" in hp:
            print(f"      Pose Variance: {hp['pose_variance']:.3f}")

    else:
        print(f"\n   HEAD POSE ANALYSIS: ERROR")

    print(f"   " + "-" * 50)


def process_dataset(fake_dir="fake", real_dir="real", output_file="headpose_results.json"):
    """
    Process dataset folders and evaluate head pose
    """

    print("\n" + "=" * 60)
    print("HEAD POSE DATASET PROCESSING")
    print("=" * 60)

    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']

    fake_videos = [
        os.path.join(fake_dir, f)
        for f in sorted(os.listdir(fake_dir))
        if any(f.lower().endswith(ext) for ext in video_extensions)
    ] if os.path.exists(fake_dir) else []

    real_videos = [
        os.path.join(real_dir, f)
        for f in sorted(os.listdir(real_dir))
        if any(f.lower().endswith(ext) for ext in video_extensions)
    ] if os.path.exists(real_dir) else []

    print(f"Found {len(fake_videos)} fake videos, {len(real_videos)} real videos")

    total_videos = len(fake_videos) + len(real_videos)
    if total_videos == 0:
        print("No videos found. Exiting...")
        return

    all_results = []

    # ======================
    # FAKE BATCH
    # ======================
    print("\n" + "=" * 60)
    print("Batch: FAKE")
    print("=" * 60)

    for video_path in fake_videos:
        filename = os.path.basename(video_path)

        print(f"\nProcessing: {filename}...")
        result = analyze_video(video_path, verbose=False)

        result["ground_truth"] = "FAKE"
        all_results.append(result)

        print_video_details(filename, result, "FAKE")

    # ======================
    # REAL BATCH
    # ======================
    print("\n" + "=" * 60)
    print("Batch: REAL")
    print("=" * 60)

    for video_path in real_videos:
        filename = os.path.basename(video_path)

        print(f"\nProcessing: {filename}...")
        result = analyze_video(video_path, verbose=False)

        result["ground_truth"] = "REAL"
        all_results.append(result)

        print_video_details(filename, result, "REAL")

    # ======================
    # METRICS
    # ======================
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    correct = sum(
        1 for r in all_results
        if r.get("prediction") == r.get("ground_truth")
    )

    total = len(all_results)
    accuracy = (correct / total * 100) if total > 0 else 0

    print(f"\nOverall Accuracy: {accuracy:.1f}% ({correct}/{total})")

    # ======================
    # SAVE JSON
    # ======================
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_videos": total,
            "accuracy": accuracy
        },
        "results": all_results
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    process_dataset(
        fake_dir="fake",
        real_dir="real",
        output_file="headpose_results.json"
    )
