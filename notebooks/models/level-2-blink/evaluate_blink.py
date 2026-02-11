import os
import json
from datetime import datetime

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

from blink_detector import analyze_blink  # pastikan blink_detector.py ada


def analyze_video(video_path, verbose=False):
    """
    Analyze single video and return blink results
    """
    results = {
        "video_path": video_path,
        "blink": None,
        "prediction": None,
        "confidence": 0
    }

    try:
        # Blink analysis
        blink_result = analyze_blink(video_path, verbose=verbose)
        if blink_result["success"]:
            results["blink"] = blink_result

            # Simple prediction based on suspicious flag
            if blink_result['suspicious']:
                results["prediction"] = "FAKE"
                results["confidence"] = 0.8
            else:
                results["prediction"] = "REAL"
                results["confidence"] = 0.9

    except Exception as e:
        results["error"] = str(e)

    return results


def print_video_details(filename, result, label):
    """
    Print blink analysis results for a single video
    """
    prediction = result.get("prediction", "ERROR")
    status = "OK" if prediction == label else "WRONG"

    print(f"\n{status} {filename}")
    print(f"   Label: {label}  |  Predicted: {prediction}")

    # Blink details
    if result.get("blink"):
        blink = result["blink"]
        print(f"\n   BLINK ANALYSIS:")
        print(f"      Status: {'SUSPICIOUS' if blink['suspicious'] else 'NORMAL'}")
        if blink['suspicious']:
            print(f"      Reasons: {', '.join(blink['reasons'])}")
        print(f"      Total Blinks: {blink['blink_count']}")
        print(f"      Blink Rate: {blink['blink_rate_per_minute']:.1f}/min")
        print(f"      Avg EAR: {blink['avg_ear']:.3f}")
        print(f"      EAR Variance: {blink['std_ear']:.3f}")
    else:
        print(f"\n   BLINK ANALYSIS: ERROR")

    print(f"   " + "-"*50)


def process_dataset(fake_dir="fake", real_dir="real", output_file="blink_results.json"):
    """
    Process all videos in fake and real directories
    """
    print("\n" + "="*60)
    print("BLINK DATASET PROCESSING")
    print("="*60)

    # Supported video formats
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']

    # Collect videos
    fake_videos = [os.path.join(fake_dir, f) for f in sorted(os.listdir(fake_dir))
                   if any(f.lower().endswith(ext) for ext in video_extensions)] if os.path.exists(fake_dir) else []
    real_videos = [os.path.join(real_dir, f) for f in sorted(os.listdir(real_dir))
                   if any(f.lower().endswith(ext) for ext in video_extensions)] if os.path.exists(real_dir) else []

    print(f"Found {len(fake_videos)} fake videos, {len(real_videos)} real videos")
    total_videos = len(fake_videos) + len(real_videos)
    if total_videos == 0:
        print("No videos found. Exiting...")
        return

    all_results = []

    # Process fake videos
    print("\n" + "="*60)
    print("Batch: FAKE")
    print("="*60)
    for video_path in fake_videos:
        filename = os.path.basename(video_path)
        print(f"\nProcessing: {filename}...")
        result = analyze_video(video_path, verbose=False)
        result["ground_truth"] = "FAKE"
        all_results.append(result)
        print_video_details(filename, result, "FAKE")

    # Process real videos
    print("\n" + "="*60)
    print("Batch: REAL")
    print("="*60)
    for video_path in real_videos:
        filename = os.path.basename(video_path)
        print(f"\nProcessing: {filename}...")
        result = analyze_video(video_path, verbose=False)
        result["ground_truth"] = "REAL"
        all_results.append(result)
        print_video_details(filename, result, "REAL")

    # Metrics
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    correct = sum(1 for r in all_results if r.get("prediction") == r.get("ground_truth"))
    total = len(all_results)
    accuracy = (correct / total * 100) if total > 0 else 0

    print(f"\nOverall Accuracy: {accuracy:.1f}% ({correct}/{total})")

    # Save results
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_videos": total,
            "accuracy": accuracy
        },
        "results": all_results
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("="*60)

    return all_results


if __name__ == "__main__":
    # Process dataset
    process_dataset(fake_dir="fake", real_dir="real", output_file="blink_results.json")
