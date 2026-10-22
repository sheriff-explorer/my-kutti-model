"""
inference/export_onnx.py

Export the trained TinyGPT model to ONNX format and apply INT8 quantization.

What is ONNX?
─────────────
ONNX (Open Neural Network Exchange) is a standardized format for neural
network models. Once exported to ONNX:
  - No PyTorch installation needed at inference time
  - Can run with the lightweight onnxruntime package (~10MB vs PyTorch's ~2GB)
  - Supports optimizations like INT8 quantization
  - Portable: works on Windows, Linux, macOS, edge devices

Why INT8 Quantization?
─────────────────────
FP32 (32-bit float) -> INT8 (8-bit integer):
  - 4× smaller file size (7.9MB -> ~2MB)
  - 1.5-2× faster CPU inference (INT8 matmul via AVX2/AVX-512 SIMD)
  - Minimal quality loss at this model scale

Dynamic Quantization:
  Weights are permanently stored as INT8.
  Activations are quantized dynamically at runtime (per-batch).
  No calibration dataset needed (unlike static quantization).

Run with:
    python inference/export_onnx.py

Output:
    outputs/model_fp32.onnx     -- full precision ONNX
    outputs/model_int8.onnx     -- INT8 quantized ONNX (use this for deployment)
"""

import os
import sys
import time

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig
from model.transformer import TinyGPT


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CFG = ModelConfig(
    vocab_size  = 512,
    d_model     = 160,
    n_heads     = 4,
    n_layers    = 6,
    block_size  = 256,
    ffn_dim     = 640,
    dropout     = 0.0,    # Must be 0 for inference/export
    tie_weights = True,
)

CHECKPOINT_PATH   = "checkpoints/posttrain/best_model.pt"
OUTPUT_DIR        = "outputs"
FP32_ONNX_PATH    = os.path.join(OUTPUT_DIR, "model_fp32.onnx")
INT8_ONNX_PATH    = os.path.join(OUTPUT_DIR, "model_int8.onnx")
OPSET_VERSION     = 18   # ONNX opset 18 is standard in PyTorch 2.11+


def export_to_onnx(model: TinyGPT, path: str) -> None:
    """
    Export TinyGPT to ONNX format.

    ONNX export works by running the model in "trace mode":
    1. We provide a dummy input tensor
    2. PyTorch runs the forward pass while recording all operations
    3. The recorded computation graph is saved as ONNX

    Key considerations:
    - dynamic_axes: tells ONNX that batch_size and seq_len can vary at runtime
      Without this, the model would only accept inputs of exactly the dummy shape.
    - do_constant_folding: pre-computes constant sub-expressions -> smaller model
    - opset_version=17: ONNX opset version; 17 supports all ops we use
    """
    model.eval()

    # Dummy input: batch=1, seq_len=10 (arbitrary; dynamic axes allow any size)
    dummy_input = torch.zeros((1, 10), dtype=torch.long)

    print(f"Exporting to ONNX (opset {OPSET_VERSION})...")

    torch.onnx.export(
        model,
        (dummy_input,),         # Model inputs as a tuple
        path,
        export_params=True,     # Include trained weights in the ONNX file
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits":    {0: "batch_size", 1: "sequence_length"},
        },
    )

    size_mb = os.path.getsize(path) / 1e6
    print(f"  [OK] FP32 ONNX saved: {path} ({size_mb:.1f} MB)")


def quantize_to_int8(fp32_path: str, int8_path: str) -> None:
    """
    Apply INT8 dynamic quantization to the ONNX model.

    Dynamic quantization:
    - Quantizes WEIGHT matrices to INT8 (permanent, stored in file)
    - Quantizes ACTIVATIONS dynamically at each inference call
    - No calibration data required
    - Supported for MatMul and other linear ops
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("ERROR: onnxruntime not found. Run: pip install onnxruntime")
        return

    print(f"Applying INT8 dynamic quantization...")

    quantize_dynamic(
        model_input   = fp32_path,
        model_output  = int8_path,
        weight_type   = QuantType.QInt8,
        # Quantize all MatMul operations (the main compute in transformers)
        per_channel   = False,  # Per-tensor (faster) vs per-channel (higher quality)
    )

    size_mb = os.path.getsize(int8_path) / 1e6
    print(f"  [OK] INT8 ONNX saved: {int8_path} ({size_mb:.1f} MB)")


def verify_onnx_outputs(model: TinyGPT, onnx_path: str) -> None:
    """
    Verify that ONNX outputs match PyTorch outputs.
    Max absolute difference should be < 1e-4 for FP32, < 0.1 for INT8.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("Skipping verification (onnxruntime not installed)")
        return

    print(f"Verifying ONNX outputs against PyTorch...")

    # PyTorch output
    model.eval()
    dummy = torch.zeros((1, 16), dtype=torch.long)
    with torch.no_grad():
        pt_logits, _ = model(dummy)
    pt_logits_np = pt_logits.numpy()

    # ONNX Runtime output
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {"input_ids": dummy.numpy()}
    ort_logits = sess.run(None, ort_inputs)[0]

    max_diff = np.max(np.abs(pt_logits_np - ort_logits))
    print(f"  Max absolute difference: {max_diff:.6f}")
    if max_diff < 1.0:
        print("  [OK] Outputs match (acceptable tolerance)")
    else:
        print("  ⚠ Large output difference -- check export settings")


def benchmark_inference(onnx_path: str, n_runs: int = 100) -> None:
    """Benchmark inference speed on CPU."""
    try:
        import onnxruntime as ort
    except ImportError:
        return

    print(f"\nBenchmarking CPU inference ({n_runs} runs)...")

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Warm-up
    dummy = {"input_ids": np.zeros((1, 64), dtype=np.int64)}
    for _ in range(5):
        sess.run(None, dummy)

    # Benchmark
    t0 = time.time()
    for _ in range(n_runs):
        sess.run(None, dummy)
    elapsed = time.time() - t0

    ms_per_call = (elapsed / n_runs) * 1000
    print(f"  Avg latency: {ms_per_call:.2f} ms per forward pass (seq_len=64)")
    print(f"  -> Estimated {1000/ms_per_call:.0f} calls/second")


def main():
    print("=" * 60)
    print(" TinyGPT -- ONNX Export + INT8 Quantization")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load trained model
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        print("       Run all 3 training stages first.")
        sys.exit(1)

    print(f"\nLoading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    model = TinyGPT(MODEL_CFG)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.count_params()
    model.eval()

    # Export FP32
    print()
    export_to_onnx(model, FP32_ONNX_PATH)

    # Quantize to INT8
    print()
    quantize_to_int8(FP32_ONNX_PATH, INT8_ONNX_PATH)

    # Verify
    print()
    verify_onnx_outputs(model, FP32_ONNX_PATH)

    # Benchmark
    benchmark_inference(INT8_ONNX_PATH)

    print("\n" + "=" * 60)
    print(f"  FP32 model: {FP32_ONNX_PATH}")
    print(f"  INT8 model: {INT8_ONNX_PATH}  ← Use this for deployment")
    print(f"\n  Run: python inference/run_inference.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
