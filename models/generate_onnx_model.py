#!/usr/bin/env python3
"""
Export Hugging Face cross-encoder models to ONNX format.

By default produces a plain FP32 graph with no post-export optimizations, suitable for CPU inference and as a baseline for inspection or fine-tuning.

Pass --fp16 and --optimization-level 2 to produce the fused FP16 variant
required for TensorRT EP and CUDA EP Tensor Core use.
"""

import argparse
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from optimum.onnxruntime import ORTModelForSequenceClassification, ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig
from transformers import AutoConfig, AutoTokenizer

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

_OPT_LEVEL_LABELS = {
    0: "none, raw ONNX graph, no post-export passes",
    1: "basic, constant folding, redundant node removal",
    2: "extended, attention / layernorm / gelu fusion",
}


@contextmanager
def stage(label: str):
    done = threading.Event()
    start = time.monotonic()

    def _spin():
        i = 0
        while not done.wait(0.1):
            elapsed = time.monotonic() - start
            print(f"\r  {_SPINNER[i % len(_SPINNER)]} {label}  [{elapsed:.0f}s]", end="", flush=True)
            i += 1

    t = threading.Thread(target=_spin, daemon=True)
    t.start()
    try:
        yield
    finally:
        done.set()
        t.join()
        elapsed = time.monotonic() - start
        print(f"\r  ✓ {label}  [{elapsed:.1f}s]")


def export_to_onnx(model_name: str, output_dir: str, task_type: str, fp16: bool, optimize_for_gpu: bool, optimization_level: int) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Model:      {model_name}")
    print(f"Output:     {output_dir}")
    print(f"Precision:  {'FP16' if fp16 else 'FP32'}")
    print(f"GPU fuse:   {'yes' if optimize_for_gpu else 'no'}")
    print(f"Opt level:  {optimization_level} ({_OPT_LEVEL_LABELS[optimization_level]})")
    print()

    total_start = time.monotonic()

    try:
        with tempfile.TemporaryDirectory() as raw_dir:
            with stage("Exporting raw ONNX graph"):
                model = ORTModelForSequenceClassification.from_pretrained(
                    model_name, export=True
                )
                model.save_pretrained(raw_dir)

            if optimization_level == 0 and not fp16:
                # Skip the optimizer entirely, it runs an internal ORT session
                # that injects com.microsoft fused ops even at level 0.
                # Copy the raw torch.onnx.export output directly.
                import shutil
                with stage("Copying raw ONNX (no optimization)"):
                    for f in sorted(Path(raw_dir).glob("*.onnx")):
                        shutil.copy2(f, output_path / f.name)
            else:
                with stage("Applying optimization pass"):
                    optimizer = ORTOptimizer.from_pretrained(raw_dir)
                    opt_config = OptimizationConfig(
                        optimization_level=optimization_level,
                        fp16=fp16,
                        optimize_for_gpu=optimize_for_gpu,
                    )
                    optimizer.optimize(
                        save_dir=str(output_path),
                        optimization_config=opt_config,
                        file_suffix="",      # writes model.onnx, not model_optimized.onnx
                    )

        with stage("Saving tokenizer and config"):
            AutoTokenizer.from_pretrained(model_name).save_pretrained(output_path)
            AutoConfig.from_pretrained(model_name).save_pretrained(output_path)

        total = time.monotonic() - total_start
        print(f"\nDone in {total:.1f}s. Files written to: {output_dir}")
        for f in sorted(output_path.glob("*")):
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  {f.name:45s} {size_mb:7.2f} MB")

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export HuggingFace cross-encoder models to ONNX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # FP32, no optimization (default, CPU inference, baseline, fine-tuning input)
  python generate_onnx_model.py \\
      -m cross-encoder/nli-deberta-v3-base \\
      -o models/base/nli-deberta-v3-base-onnx/

  # FP16 + fused, required for TensorRT EP and CUDA EP Tensor Core use
  python generate_onnx_model.py \\
      -m cross-encoder/nli-deberta-v3-base \\
      -o models/fp16_fused/nli-deberta-v3-base-onnx/ \\
      --fp16 --optimization-level 2

  # FP16 + fused, DeBERTa-v3, use level 1 to avoid attention-fusion crashes
  python generate_onnx_model.py \\
      -m cross-encoder/nli-deberta-v3-base \\
      -o models/fp16_fused/nli-deberta-v3-base-onnx/ \\
      --fp16 --optimization-level 1
        """,
    )
    parser.add_argument(
        "-m", "--model",
        required=True,
        help='HuggingFace model ID (e.g. "cross-encoder/nli-deberta-v3-base")',
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory for model.onnx and tokenizer files",
    )
    parser.add_argument(
        "-t", "--task",
        default="text-classification",
        help="Task type (default: text-classification)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Cast weights to FP16. Required for TensorRT EP and CUDA Tensor Core use.",
    )
    parser.add_argument(
        "--optimize-for-gpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Apply GPU-specific graph fusions during the optimization pass "
            "(default: on when --fp16 is set, off otherwise). "
            "Pass --no-optimize-for-gpu to keep FP16 weights without GPU-specific fusions, "
            "useful for large models where level-2 GPU fusion passes crash (e.g. DeBERTa-v3)."
        ),
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help=(
            "ONNX Runtime graph optimization level (default: 0, no optimization). "
            "1: basic (constant folding, redundant node removal). "
            "2: extended (attention / layernorm / gelu fusion). "
            "Use 1 instead of 2 for models with non-standard attention (e.g. DeBERTa-v3) "
            "where level-2 attention fusion passes can crash."
        ),
    )
    args = parser.parse_args()

    optimize_for_gpu = args.optimize_for_gpu if args.optimize_for_gpu is not None else args.fp16

    export_to_onnx(
        model_name=args.model,
        output_dir=args.output,
        task_type=args.task,
        fp16=args.fp16,
        optimize_for_gpu=optimize_for_gpu,
        optimization_level=args.optimization_level,
    )


if __name__ == "__main__":
    main()
