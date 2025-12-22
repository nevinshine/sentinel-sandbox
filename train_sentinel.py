# train_sentinel.py

import torch
import torch.nn as nn
import torch.optim as optim

from data.sentinel_bridge import SentinelBridge
from models.dwn_classifier import DWNClassifier


def main():
    device = torch.device("cpu")  # stay CPU-only

    # -----------------------------
    # 1. Load Sentinel Data
    # -----------------------------
    bridge = SentinelBridge(
        window_size=100,
        max_syscall=335,
        thermometer_resolution=8
    )

    x, y = bridge.process_log("sentinel_log.csv")

    if x is None:
        print("❌ No data. Exiting.")
        return

    x = x.to(device)
    y = y.to(device)

    print(f"📊 Data Shape: X={x.shape}, y={y.shape}")

    # -----------------------------
    # 2. Model
    # -----------------------------
    model = DWNClassifier(
        num_inputs=x.shape[1],
        tuple_size=4,
        num_classes=2
    ).to(device)

    # -----------------------------
    # 3. Optimizer & Loss
    # -----------------------------
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # -----------------------------
    # 4. Training Loop (Sanity)
    # -----------------------------
    epochs = 5

    print("🚀 Starting Sentinel DWN training (Normal-only)...")

    for epoch in range(1, epochs + 1):
        model.train()

        optimizer.zero_grad()

        outputs = model(x)              # [Batch, 2]
        loss = criterion(outputs, y)    # y = 0 (Normal)

        loss.backward()
        optimizer.step()

        # -------------------------
        # Diagnostics (IMPORTANT)
        # -------------------------
        with torch.no_grad():
            normal_score = outputs[:, 0].mean().item()
            attack_score = outputs[:, 1].mean().item()

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {loss.item():.4f} | "
            f"Normal Score: {normal_score:.3f} | "
            f"Attack Score: {attack_score:.3f}"
        )

    print("✅ Training completed without crashing.")


if __name__ == "__main__":
    main()
