import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "research" / "EXP-06" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 1. Training Curves
history_path = ROOT / "models_storage_v2" / "RELIANCE_NS" / "history.csv"
history = pd.read_csv(history_path)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
epochs = range(1, len(history) + 1)
ax.plot(epochs, history["loss"], label="Train Loss (MSE)", color="#1f77b4", linewidth=2)
ax.plot(epochs, history["val_loss"], label="Validation Loss (MSE)", color="#ff7f0e", linewidth=2)
ax.set_title("V2 GRU Challenger Training Progress (Fold 3)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Loss (Scaled MSE)", fontsize=11)
ax.legend(frameon=True, facecolor="white", framealpha=0.9)
ax.set_xticks(epochs)
plt.tight_layout()
fig1_path = FIG_DIR / "1_training_curves.png"
fig.savefig(fig1_path)
plt.close(fig)
print(f"Saved: {fig1_path}")

# 2. Forecast Comparison
preds_path = ROOT / "models_storage_v2" / "RELIANCE_NS" / "predictions.csv"
preds = pd.read_csv(preds_path)
preds["target_date"] = pd.to_datetime(preds["target_date"])

fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
ax.plot(preds["target_date"], preds["actual_close"], label="Actual Price (INR)", color="#1f77b4", linewidth=2)
ax.plot(preds["target_date"], preds["predicted_close"], label="Predicted Price (INR)", color="#e15759", linestyle="--", linewidth=1.8)
ax.set_title("RELIANCE.NS Stock Price Prediction (Exploratory Fold 3 Test Window)", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Target Session Date", fontsize=11)
ax.set_ylabel("Price (INR)", fontsize=11)
ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper left")
plt.xticks(rotation=30)
plt.tight_layout()
fig2_path = FIG_DIR / "2_forecast_comparison.png"
fig.savefig(fig2_path)
plt.close(fig)
print(f"Saved: {fig2_path}")

# 3. Directional Evaluation
metadata_path = ROOT / "models_storage_v2" / "RELIANCE_NS" / "metadata.json"
meta = json.loads(metadata_path.read_text(encoding="utf-8"))
metrics = meta["metrics"]
cm = np.array(metrics["confusion_matrix"])
# Extract 2x2 binary UP/DOWN (excluding FLAT)
# class_order: ["DOWN", "FLAT", "UP"]
cm_2x2 = np.array([[cm[0][0], cm[0][2]], [cm[2][0], cm[2][2]]])

fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
cax = ax.matshow(cm_2x2, cmap="Blues", alpha=0.85)

for i in range(2):
    for j in range(2):
        val = cm_2x2[i, j]
        ax.text(j, i, f"{val}", ha="center", va="center", fontsize=14, fontweight="bold",
                color="white" if val > cm_2x2.max() / 2 else "black")

fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred DOWN", "Pred UP"], fontsize=11)
ax.set_yticklabels(["Actual DOWN", "Actual UP"], fontsize=11)
title_text = f"Directional Confusion Matrix (Fold 3)\nAccuracy: {metrics['directional_accuracy']*100:.1f}% | MCC: {metrics['mcc']:+.4f} | UP Recall: {metrics['recall'][2]*100:.1f}%"
ax.set_title(title_text, fontsize=12, fontweight="bold", pad=20)
plt.tight_layout()
fig3_path = FIG_DIR / "3_directional_evaluation.png"
fig.savefig(fig3_path)
plt.close(fig)
print(f"Saved: {fig3_path}")

# 4. Model Leaderboard
lb_path = ROOT / "research" / "EXP-06" / "artifacts" / "exploratory_run" / "leaderboard_summary.csv"
lb = pd.read_csv(lb_path)
summary = lb.groupby("model").agg({
    "price_mae": "mean",
    "price_rmse": "mean",
    "dir_acc": "mean",
    "mcc": "mean"
}).loc[["gru", "naive", "lstm", "cnn_lstm", "ridge"]]

models_clean = ["GRU", "Naive", "LSTM", "CNN-LSTM", "Ridge"]
maes = summary["price_mae"].values
dir_accs = summary["dir_acc"].values * 100
mccs = summary["mcc"].values

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

# Bar chart 1: Price MAE
bars1 = ax1.bar(models_clean, maes, color=["#2ca02c", "#7f7f7f", "#1f77b4", "#9467bd", "#d62728"], width=0.55)
ax1.set_title("Cross-Fold Mean Price MAE (Lower is Better)", fontsize=11, fontweight="bold")
ax1.set_ylabel("Price MAE (INR)", fontsize=10)
ax1.set_ylim(10, 15)
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# Bar chart 2: Directional Accuracy (%)
bars2 = ax2.bar(models_clean, dir_accs, color=["#2ca02c", "#7f7f7f", "#1f77b4", "#9467bd", "#d62728"], width=0.55)
ax2.axhline(50, color="red", linestyle="--", linewidth=1, label="Chance (50%)")
ax2.set_title("Cross-Fold Mean Directional Accuracy", fontsize=11, fontweight="bold")
ax2.set_ylabel("Accuracy (%)", fontsize=10)
ax2.set_ylim(0, 60)
ax2.legend(loc="lower right")
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
fig4_path = FIG_DIR / "4_model_leaderboard.png"
fig.savefig(fig4_path)
plt.close(fig)
print(f"Saved: {fig4_path}")
