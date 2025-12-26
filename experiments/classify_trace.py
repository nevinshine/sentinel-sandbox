import json
import torch
import numpy as np

from data.sentinel_bridge import SentinelBridge
from models.dwn_classifier import DWNClassifier

MODEL_PATH = "sentinel_model.pt"
THRESH_PATH = "checkpoints/thresholds.json"
TRACE_PATH = "sentinel_log.csv"

def classify(score, t):
    if score < t["critical"]:
        return "CRITICAL"
    elif score < t["anomalous"]:
        return "ANOMALOUS"
    elif score < t["suspicious"]:
        return "SUSPICIOUS"
    else:
        return "NORMAL"

def main():
    print("[*] Loading thresholds...")
    with open(THRESH_PATH) as f:
        thresholds = json.load(f)

    print("[*] Loading Sentinel Bridge...")
    bridge = SentinelBridge()

    x, _ = bridge.process_log(TRACE_PATH)
    if x is None:
        print("No data")
        return

    print(f"[+] Loaded {len(x)} syscall windows")

    print("[*] Loading DWN model...")
    model = DWNClassifier(num_inputs=x.shape[1])
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        scores = model(x)
        anomaly_scores = scores[:, 0] - scores[:, 1]

    print("\n=== WINDOW CLASSIFICATION ===")
    counts = {"NORMAL": 0, "SUSPICIOUS": 0, "ANOMALOUS": 0, "CRITICAL": 0}

    for s in anomaly_scores.numpy():
        label = classify(s, thresholds)
        counts[label] += 1

    for k, v in counts.items():
        print(f"{k:10s}: {v}")

if __name__ == "__main__":
    main()
