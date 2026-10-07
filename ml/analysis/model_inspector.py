import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import tensorflow as tf
from tensorflow.keras.models import Model

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_ANALYSIS_DIR = PROJECT_ROOT / "experiments" / "model_analysis"


def inspect_model_architecture(model: Model, symbol: str = "RELIANCE_NS", save_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Exhaustive mathematical model inspection tool.
    Extracts layer shapes, tensor dimensions, and trainable parameter counts.
    Saves report to experiments/model_analysis/<SYMBOL>/model_summary.txt.
    """
    target_dir = save_dir or (MODEL_ANALYSIS_DIR / symbol.replace(".", "_"))
    target_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append(f"  STOCKSENSE AI — NEURAL NETWORK ARCHITECTURE INSPECTION REPORT")
    summary_lines.append(f"  Target Equity: {symbol}")
    summary_lines.append("=" * 80)

    # 1. Input Shape
    input_shape = model.input_shape
    summary_lines.append(f"\n[1] Input Tensor Specification:")
    summary_lines.append(f"    - Input Shape: {input_shape} (Batch Size x 30 Timesteps x 6 Features)")
    summary_lines.append(f"    - Feature Channels: [Open, High, Low, Close, Volume, MA20]")

    # 2. Layer-by-Layer Inspection
    summary_lines.append(f"\n[2] Layer-by-Layer Mathematical Breakdown:")
    summary_lines.append(f"    {'Index':<6} {'Layer Name':<20} {'Layer Type':<18} {'Output Shape':<24} {'Param #':<10}")
    summary_lines.append("    " + "-" * 76)

    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    detailed_layer_params = {}

    for idx, layer in enumerate(model.layers):
        p_count = layer.count_params()
        total_params += p_count
        if layer.trainable:
            trainable_params += p_count
        else:
            non_trainable_params += p_count

        out_shape = str(layer.output_shape)
        summary_lines.append(
            f"    {idx:<6} {layer.name:<20} {layer.__class__.__name__:<18} {out_shape:<24} {p_count:<10}"
        )

        weights = layer.get_weights()
        if weights:
            shapes = [w.shape for w in weights]
            detailed_layer_params[layer.name] = {
                "class": layer.__class__.__name__,
                "weight_shapes": shapes,
                "param_count": p_count
            }

    # 3. Dedicated Parameter Geometry
    summary_lines.append(f"\n[3] Kernel & Parameter Tensor Geometry:")
    for layer_name, info in detailed_layer_params.items():
        summary_lines.append(f"    Layer: '{layer_name}' ({info['class']}) — Params: {info['param_count']}")
        for w_idx, shape in enumerate(info["weight_shapes"]):
            desc = "Weight/Kernel" if w_idx == 0 else ("Recurrent Kernel" if w_idx == 1 and info["class"] == "LSTM" else "Bias")
            summary_lines.append(f"      - Component {w_idx} ({desc}): shape = {shape}")

    # 4. Global Counts
    summary_lines.append(f"\n[4] Parameter Count Totals:")
    summary_lines.append(f"    - Total Parameters:         {total_params:,}")
    summary_lines.append(f"    - Trainable Parameters:     {trainable_params:,}")
    summary_lines.append(f"    - Non-Trainable Parameters: {non_trainable_params:,}")
    summary_lines.append("=" * 80 + "\n")

    report_text = "\n".join(summary_lines)

    # Save to disk
    report_file = target_dir / "model_summary.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    return {
        "symbol": symbol,
        "input_shape": input_shape,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "detailed_layer_params": detailed_layer_params,
        "report_path": str(report_file),
        "report_text": report_text
    }
