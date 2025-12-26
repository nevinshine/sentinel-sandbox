import json
import torch
import numpy as np

from data.sentinel_bridge import SentinelBridge
from models.dwn_classifier import DWNClassifier

MODEL_PATH = "sentinel_model.pt"
TRACE_PATH = "sentinel_log.csv"
OUT_PATH = "checkpoints/thresholds.json"


def main():
    print("[*] Loading Sentinel Bridge...")
    bridge = SentinelBridge(
        window_size=100,
        thermometer_resolution=8,
        num_buckets=4
    )

    X, _ = bridge.process_log(TRACE_PATH)
    if X is None:
        print("No data. Exiting.")
        return

    print(f"[+] Loaded {len(X)} normal syscall windows")

    print("[*] Loading trained DWN model...")
    model = DWNClassifier(num_inputs=X.shape[1])
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        scores = model(X)
        normal = scores[:, 0]
        attack = scores[:, 1]
        anomaly = normal - attack

    anomaly_np = anomaly.numpy()

    mu = float(np.mean(anomaly_np))
    sigma = float(np.std(anomaly_np))

    thresholds = {
        "mean": mu,
        "std": sigma,
        "suspicious": mu - sigma,
        "anomalous": mu - 2 * sigma,
        "critical": mu - 3 * sigma,
    }

    print("\n=== NORMAL BASELINE ===")
    for k, v in thresholds.items():
        print(f"{k:>12}: {v:.4f}")

    with open(OUT_PATH, "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"\n[✓] Thresholds saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
