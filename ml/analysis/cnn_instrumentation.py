import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

# Set non-interactive matplotlib backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Model

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_ANALYSIS_DIR = PROJECT_ROOT / "experiments" / "model_analysis"


class CNNInstrumentation:
    """
    Mathematical Instrumentation for Conv1D Layer in Hybrid CNN-LSTM.
    
    Capabilities:
    1. Kernel & Bias Extraction: Extracts and archives learned 1D convolution weights.
    2. Activation Mapping: Computes intermediate activations across the 30-day temporal sequence.
    3. Strongest Activation Profiling: Identifies which temporal steps trigger peak filter responses.
    4. Visualization: Renders activation heatmaps correlated with time steps.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.base_output_dir = output_dir or MODEL_ANALYSIS_DIR

    def get_stock_cnn_dir(self, symbol: str) -> Path:
        clean = symbol.replace(".", "_")
        cnn_dir = self.base_output_dir / clean / "cnn"
        cnn_dir.mkdir(parents=True, exist_ok=True)
        return cnn_dir

    def extract_conv1d_weights(self, model: Model, symbol: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts Conv1D kernel weights and biases.
        Kernel shape: (kernel_size=3, in_channels=6, filters=64)
        Bias shape: (filters=64,)
        Saves:
            experiments/model_analysis/<SYMBOL>/cnn/kernels.npy
            experiments/model_analysis/<SYMBOL>/cnn/biases.npy
        """
        cnn_dir = self.get_stock_cnn_dir(symbol)
        conv_layer = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv1D) or "conv1d" in layer.name.lower():
                conv_layer = layer
                break

        if conv_layer is None:
            raise ValueError("No Conv1D layer identified in model architecture.")

        weights = conv_layer.get_weights()
        if len(weights) < 2:
            raise ValueError(f"Expected kernel and bias in Conv1D, found {len(weights)} weight tensors.")

        kernels = weights[0]  # Shape: (3, 6, 64)
        biases = weights[1]   # Shape: (64,)

        # Save numpy arrays
        np.save(cnn_dir / "kernels.npy", kernels)
        np.save(cnn_dir / "biases.npy", biases)
        logger.info(f"[{symbol}] Saved Conv1D kernels {kernels.shape} and biases {biases.shape} to {cnn_dir}")

        return kernels, biases

    def extract_activations(
        self,
        model: Model,
        X_sample: np.ndarray,
        symbol: str,
        input_dates: Optional[List[str]] = None,
        sample_index: int = 0
    ) -> Dict[str, Any]:
        """
        Passes a sample sequence (30, 6) or batch of samples through the Conv1D layer
        and computes the full activation map (30, 64).
        
        Outputs:
            - strongest_activations.csv: per-filter max activation and timestep
            - activation_maps.png: 2D heatmap of 30 timesteps vs 64 filters
        """
        cnn_dir = self.get_stock_cnn_dir(symbol)

        conv_layer = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv1D) or "conv1d" in layer.name.lower():
                conv_layer = layer
                break

        if conv_layer is None:
            raise ValueError("No Conv1D layer identified in model.")

        # Build submodel that outputs Conv1D activations
        activation_model = Model(inputs=model.inputs, outputs=conv_layer.output)

        if X_sample.ndim == 2:
            X_batch = np.expand_dims(X_sample, axis=0)
        else:
            X_batch = X_sample

        # Predict activations: shape (Batch, 30, 64)
        activations = activation_model.predict(X_batch, verbose=0)
        sample_act = activations[sample_index]  # Shape: (30, 64)

        num_timesteps, num_filters = sample_act.shape

        # Identify strongest activations per filter
        records = []
        for f_idx in range(num_filters):
            filter_series = sample_act[:, f_idx]
            max_act = float(np.max(filter_series))
            strongest_step = int(np.argmax(filter_series))
            mean_act = float(np.mean(filter_series))

            date_str = input_dates[strongest_step] if input_dates and strongest_step < len(input_dates) else f"Step_{strongest_step}"
            
            # Receptive field for kernel_size=3 with same padding: [t-1, t, t+1]
            rf_start = max(0, strongest_step - 1)
            rf_end = min(num_timesteps - 1, strongest_step + 1)
            if input_dates and rf_end < len(input_dates):
                rf_window = f"{input_dates[rf_start]} to {input_dates[rf_end]}"
            else:
                rf_window = f"Step_{rf_start} to Step_{rf_end}"

            records.append({
                "filter_index": f_idx,
                "max_activation": round(max_act, 5),
                "peak_timestep": strongest_step,
                "peak_center_date": date_str,
                "correlated_date": date_str,
                "receptive_field_window": rf_window,
                "receptive_field_steps": f"[{rf_start}, {strongest_step}, {rf_end}]",
                "mean_activation": round(mean_act, 5)
            })

        df_act = pd.DataFrame(records).sort_values("max_activation", ascending=False).reset_index(drop=True)
        csv_path = cnn_dir / "strongest_activations.csv"
        df_act.to_csv(csv_path, index=False)

        # -------------------------------------------------------------------
        # Render Activation Heatmap Plot
        # -------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        # Heatmap: rows = 30 timesteps, columns = 64 filters
        sns.heatmap(
            sample_act,
            cmap="viridis",
            cbar=True,
            ax=ax,
            yticklabels=input_dates if (input_dates and len(input_dates) == num_timesteps) else True
        )
        ax.set_title(f"{symbol} — Conv1D Layer Feature Activations (Sample #{sample_index})", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Filter Index (0 to 63)", fontsize=10)
        ax.set_ylabel("Temporal Timestep (30-day Window)", fontsize=10)
        fig.tight_layout()

        plot_path = cnn_dir / "activation_maps.png"
        fig.savefig(plot_path)
        plt.close(fig)

        logger.info(f"[{symbol}] Saved activation profile and heatmap to {cnn_dir}")

        return {
            "symbol": symbol,
            "sample_index": sample_index,
            "activation_shape": list(sample_act.shape),
            "csv_path": str(csv_path),
            "plot_path": str(plot_path),
            "top_filters": df_act.head(5).to_dict(orient="records")
        }
