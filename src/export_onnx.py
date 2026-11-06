"""Export and validate the best checkpoint as a CPU-ready ONNX model."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import get_classes, load_config  # noqa: E402
from evaluate import _Sample, _load_model, _read_test_samples  # noqa: E402


class _NormWrapper(nn.Module):
    """Apply ImageNet normalization inside the ONNX graph."""

    def __init__(self, backbone: nn.Module) -> None:
        """Store the backbone and register its RGB normalization constants."""
        super().__init__()
        self.backbone = backbone
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize an RGB [0, 1] batch and return model logits."""
        return self.backbone((images - self.mean) / self.std)


class _RawImageDataset(Dataset[tuple[torch.Tensor, int]]):
    """Load unnormalized RGB tensors to feed the ONNX graph."""

    def __init__(self, samples: list[_Sample], image_size: int) -> None:
        """Store test samples and a resize/crop transform that returns [0, 1]."""
        self.samples = samples
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        """Return the number of images in the manifest."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Load a sample as an unnormalized RGB tensor."""
        sample = self.samples[index]
        with Image.open(sample.path) as source_image:
            image = source_image.convert("RGB")
        return self.transform(image), sample.label


def _export(wrapped: nn.Module, output_path: Path, image_size: int) -> None:
    """Export a CPU model with opset 17 and a dynamic batch dimension."""
    wrapped = wrapped.cpu().eval()
    dummy = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            wrapped,
            (dummy,),
            str(output_path),
            opset_version=17,
            input_names=["images"],
            output_names=["logits"],
            dynamic_axes={
                "images": {0: "batch"},
                "logits": {0: "batch"},
            },
            do_constant_folding=True,
            dynamo=False,
        )


def _make_session(
    onnx_path: Path,
    provider: str = "CPUExecutionProvider",
) -> ort.InferenceSession:
    """Create an ONNX Runtime session using the requested execution provider."""
    return ort.InferenceSession(str(onnx_path), providers=[provider])


def _session_logits(
    session: ort.InferenceSession,
    images: np.ndarray,
) -> np.ndarray:
    """Run one batch through ONNX Runtime and return its logits."""
    input_name = session.get_inputs()[0].name
    return cast(np.ndarray, session.run(None, {input_name: images})[0])


def _verify_own_images(
    onnx_path: Path,
    wrapped: nn.Module,
    raw_samples: list[_Sample],
    image_size: int,
) -> tuple[ort.InferenceSession, list[np.ndarray]]:
    """Check ONNX CPU logits against PyTorch on 20 own-camera test images."""
    own_samples = [sample for sample in raw_samples if sample.source == "own"]
    if len(own_samples) < 20:
        raise ValueError(
            "ONNX validation requires 20 own-camera test images; "
            f"the manifest has {len(own_samples)}."
        )
    own_dataset = _RawImageDataset(own_samples[:20], image_size)
    loader = DataLoader(own_dataset, batch_size=1, shuffle=False, num_workers=0)
    session = _make_session(onnx_path)
    wrapped = wrapped.cpu().eval()
    max_difference = 0.0
    inputs: list[np.ndarray] = []
    with torch.inference_mode():
        for images, _ in loader:
            image_array = images.numpy()
            torch_logits = wrapped(images).numpy()
            onnx_logits = _session_logits(session, image_array)
            max_difference = max(
                max_difference,
                float(np.max(np.abs(torch_logits - onnx_logits))),
            )
            inputs.append(image_array)
    print(
        "CPU ONNX/PyTorch match on 20 own-camera test images: "
        f"max |difference|={max_difference:.2e}"
    )
    if max_difference > 1e-4:
        raise RuntimeError(
            f"ONNX Runtime CPU diverges from PyTorch: {max_difference:.2e} > 1e-4"
        )
    print("  Match check passed (tolerance 1e-4).")
    return session, inputs


def _benchmark(
    session: ort.InferenceSession,
    image: np.ndarray,
    label: str,
    runs: int = 100,
) -> float:
    """Benchmark a single image and print mean and 95th percentile latency."""
    for _ in range(5):
        _session_logits(session, image)
    elapsed_ms: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        _session_logits(session, image)
        elapsed_ms.append((time.perf_counter() - started) * 1000)
    values = np.asarray(elapsed_ms)
    mean_ms = float(values.mean())
    p95_ms = float(np.percentile(values, 95))
    print(
        f"{label} latency ({runs} runs): "
        f"mean={mean_ms:.2f} ms, p95={p95_ms:.2f} ms"
    )
    return mean_ms


def _evaluate_macro_f1(
    wrapped: nn.Module,
    samples: list[_Sample],
    classes: list[str],
    image_size: int,
) -> float:
    """Evaluate the candidate checkpoint's macro F1 on the whole test split."""
    dataset = _RawImageDataset(samples, image_size)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    labels: list[int] = []
    predictions: list[int] = []
    wrapped = wrapped.cpu().eval()
    with torch.inference_mode():
        for images, batch_labels in loader:
            logits = wrapped(images)
            predictions.extend(logits.argmax(dim=1).tolist())
            labels.extend(batch_labels.tolist())
    from sklearn.metrics import f1_score

    return float(
        f1_score(
            labels,
            predictions,
            labels=list(range(len(classes))),
            average="macro",
            zero_division=0,
        )
    )


def _previous_onnx_macro_f1(
    onnx_path: Path,
    samples: list[_Sample],
    classes: list[str],
    image_size: int,
) -> float:
    """Measure an existing ONNX model on the same complete test split."""
    from sklearn.metrics import f1_score

    dataset = _RawImageDataset(samples, image_size)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    session = _make_session(onnx_path)
    labels: list[int] = []
    predictions: list[int] = []
    for images, batch_labels in loader:
        logits = _session_logits(session, images.numpy())
        predictions.extend(logits.argmax(axis=1).tolist())
        labels.extend(batch_labels.tolist())
    return float(
        f1_score(
            labels,
            predictions,
            labels=list(range(len(classes))),
            average="macro",
            zero_division=0,
        )
    )


def _backup_outputs(onnx_path: Path, labels_path: Path) -> Path:
    """Copy current export outputs into a new dated backup directory."""
    backup_dir = onnx_path.parent / f"_backup_{datetime.now():%Y%m%d_%H%M%S}"
    if backup_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing backup: {backup_dir}")
    backup_dir.mkdir(parents=True)
    if onnx_path.is_file():
        shutil.copy2(onnx_path, backup_dir / onnx_path.name)
    external_data_path = onnx_path.with_name(f"{onnx_path.name}.data")
    if external_data_path.is_file():
        shutil.copy2(external_data_path, backup_dir / external_data_path.name)
    if labels_path.is_file():
        shutil.copy2(labels_path, backup_dir / labels_path.name)
    print(f"Previous export files backed up to {backup_dir}")
    return backup_dir


def _export_cli(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse export command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow replacing the ONNX model if test macro F1 regresses",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Export the checkpoint, enforce the regression guard, and benchmark it."""
    args = _export_cli(argv)
    config = load_config()
    classes = get_classes()
    models_dir = ROOT / "models"
    checkpoint_path = models_dir / "best.pt"
    onnx_path = ROOT / config["paths"]["onnx"]
    labels_path = ROOT / config["paths"]["labels"]
    if not onnx_path.is_absolute():
        onnx_path = ROOT / onnx_path
    if not labels_path.is_absolute():
        labels_path = ROOT / labels_path

    print(f"Loading checkpoint: {checkpoint_path}")
    backbone, checkpoint_classes, model_name, image_size = _load_model(
        checkpoint_path,
        classes,
    )
    if checkpoint_classes != classes:
        raise ValueError(
            f"Checkpoint classes {checkpoint_classes} do not match config {classes}"
        )
    wrapped = _NormWrapper(backbone.cpu()).cpu().eval()
    samples = _read_test_samples(ROOT / config["paths"]["processed_root"], classes)

    candidate_f1 = _evaluate_macro_f1(wrapped, samples, classes, image_size)
    print(f"Candidate checkpoint whole-test macro F1: {candidate_f1:.4f}")
    previous_f1: float | None = None
    if onnx_path.is_file():
        try:
            previous_f1 = _previous_onnx_macro_f1(
                onnx_path,
                samples,
                classes,
                image_size,
            )
            print(
                f"Previous ONNX whole-test macro F1: {float(previous_f1):.4f}; "
                f"candidate checkpoint: {float(candidate_f1):.4f}"
            )
        except Exception as error:
            if not args.force:
                raise RuntimeError(
                    "Could not evaluate the previous ONNX model for the "
                    "regression guard; use --force to override."
                ) from error
            print(
                f"Could not evaluate previous ONNX for comparison ({error}); "
                "--force permits replacement."
            )
    else:
        print("No previous ONNX model exists; regression comparison skipped.")
    if (
        previous_f1 is not None
        and candidate_f1 < previous_f1
        and not args.force
    ):
        raise RuntimeError(
            "Refusing to replace models/model.onnx: whole-test macro F1 "
            f"would decrease from {previous_f1:.4f} to {candidate_f1:.4f}. "
            "Pass --force to override."
        )
    if previous_f1 is not None and candidate_f1 < previous_f1:
        print(
            f"Regression override: previous ONNX macro F1={previous_f1:.4f}, "
            f"candidate checkpoint macro F1={candidate_f1:.4f}."
        )

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = onnx_path.with_name(
        f".{onnx_path.stem}.exporting.onnx"
    )
    try:
        print(
            f"Exporting {model_name} to {temporary_path} "
            "(opset 17, dynamic batch, CPU)"
        )
        _export(wrapped, temporary_path, image_size)
        onnx.checker.check_model(str(temporary_path))
        print(
            f"ONNX graph valid ({temporary_path.stat().st_size / 1_048_576:.2f} MiB)"
        )
        cpu_session, own_inputs = _verify_own_images(
            temporary_path,
            wrapped,
            samples,
            image_size,
        )
        cpu_mean_ms = _benchmark(
            cpu_session,
            own_inputs[0],
            "CPU ONNX Runtime",
            runs=100,
        )

        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available_providers:
            print(
                "ONNX Runtime CUDA provider unavailable; "
                "skipping GPU parity and speed comparison."
            )
        else:
            try:
                cuda_session = _make_session(
                    temporary_path,
                    "CUDAExecutionProvider",
                )
                max_difference = 0.0
                for sample_input in own_inputs:
                    cpu_logits = _session_logits(cpu_session, sample_input)
                    cuda_logits = _session_logits(cuda_session, sample_input)
                    max_difference = max(
                        max_difference,
                        float(np.max(np.abs(cpu_logits - cuda_logits))),
                    )
                print(
                    "CUDA ONNX Runtime check on 20 own-camera images: "
                    f"max |CPU-CUDA logits|={max_difference:.2e}"
                )
                cuda_mean_ms = _benchmark(
                    cuda_session,
                    own_inputs[0],
                    "CUDA ONNX Runtime",
                    runs=100,
                )
                if cuda_mean_ms > 0:
                    speed_difference = (
                        (cpu_mean_ms - cuda_mean_ms) / cpu_mean_ms * 100
                    )
                    print(
                        f"CUDA mean latency speed difference vs CPU: "
                        f"{speed_difference:+.1f}% "
                        "(positive means CUDA is faster)."
                    )
            except Exception as error:
                print(
                    f"ONNX Runtime CUDA provider is installed but unavailable "
                    f"for this export: {error}"
                )

        _backup_outputs(onnx_path, labels_path)
        temporary_path.replace(onnx_path)
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        labels_path.write_text(
            json.dumps(classes, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    print(f"Exported ONNX model: {onnx_path}")
    print(f"Labels: {labels_path}")


if __name__ == "__main__":
    main()
