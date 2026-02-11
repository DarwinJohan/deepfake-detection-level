import subprocess
import json
from concurrent.futures import ThreadPoolExecutor

# ==========================
# PATH INTERPRETER
# ==========================
PY_GLOBAL = "python"                     # Python global untuk emotion
PY_VENV = r"venv\Scripts\python.exe"     # Python venv untuk blink/headpose/texture

# ==========================
# DETECTOR SCRIPTS
# ==========================
SCRIPTS = {
    "emotion": (PY_GLOBAL, "evaluate_emotion.py"),
    "blink": (PY_VENV, "evaluate_blink.py"),
    "headpose": (PY_VENV, "evaluate_headpose.py"),
    "texture": (PY_VENV, "evaluate_texture.py"),
}

# ==========================
# HELPER FUNCTION
# ==========================
def run_detector(name, python_exec, script):
    """Jalankan script detector dan baca JSON output"""
    try:
        result = subprocess.run(
            [python_exec, script],
            capture_output=True,
            text=True,
            check=True
        )
        # Asumsi setiap script save JSON sendiri
        # Kita bisa parse stdout jika script print JSON juga
        return name, json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {name} failed:", e.stderr)
        return name, None
    except json.JSONDecodeError:
        # fallback: baca dari file JSON output
        if name == "emotion":
            filename = "emotion_evaluation.json"
        else:
            filename = f"{name}_evaluation.json"
        try:
            with open(filename) as f:
                return name, json.load(f)
        except:
            return name, None

# ==========================
# MAIN
# ==========================
def main():
    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(run_detector, name, py, scr)
            for name, (py, scr) in SCRIPTS.items()
        ]
        for f in futures:
            name, res = f.result()
            results[name] = res

    print("\n=== FINAL RESULTS ===")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
