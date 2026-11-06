"""Evaluate waste-sorter checkpoints using the processed test manifest."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import NamedTuple, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import load_config  # noqa: E402


class _Sample(NamedTuple):
    """One image path, true class index, and manifest source."""

    path: Path
    label: int
    source: str


class _Predictions(NamedTuple):
    """Predictions and metadata for a set of test images."""

    labels: np.ndarray
    preds: np.ndarray
    probs: np.ndarray
    paths: list[str]
    sources: list[str]


class _ManifestDataset(Dataset[tuple[torch.Tensor, int]]):
    """Load test images with class and source metadata from the manifest."""

    def __init__(
        self,
        samples: list[_Sample],
        transform: transforms.Compose,
    ) -> None:
        """Store samples and the deterministic evaluation transform."""
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of test images."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Load and transform an image, returning its tensor and class index."""
        sample = self.samples[index]
        with Image.open(sample.path) as source_image:
            image = source_image.convert("RGB")
        return self.transform(image), sample.label


def _val_transform(image_size: int = 224) -> transforms.Compose:
    """Create the deterministic ImageNet-normalized evaluation transform."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )


def _build_model(arch: str, num_classes: int) -> nn.Module:
    """Create a supported torchvision model with a replacement final layer."""
    builders = {
        "efficientnet_b0": models.efficientnet_b0,
        "mobilenet_v3_large": models.mobilenet_v3_large,
        "mobilenet_v3_small": models.mobilenet_v3_small,
        "convnext_tiny": models.convnext_tiny,
    }
    if arch not in builders:
        raise ValueError(f"Unknown model name in checkpoint: {arch!r}")
    model = builders[arch](weights=None)
    final_layer = model.classifier[-1]
    if not isinstance(final_layer, nn.Linear):
        raise TypeError(f"Unexpected classifier structure for {arch!r}")
    model.classifier[-1] = nn.Linear(final_layer.in_features, num_classes)
    return model


def _load_model(
    ckpt_path: Path,
    expected_classes: list[str] | None = None,
) -> tuple[nn.Module, list[str], str, int]:
    """Load a checkpoint and return its model, labels, name, and image size."""
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model_name = checkpoint.get("model_name", checkpoint.get("arch"))
    if not isinstance(model_name, str):
        raise ValueError(f"Checkpoint has no model name: {ckpt_path}")
    classes = checkpoint.get("classes")
    if not isinstance(classes, list) or not all(
        isinstance(label, str) for label in classes
    ):
        raise ValueError(f"Checkpoint has no valid class list: {ckpt_path}")
    if expected_classes is not None and classes != expected_classes:
        raise ValueError(
            f"Checkpoint classes {classes} do not match configured classes "
            f"{expected_classes}"
        )
    image_size = int(checkpoint.get("image_size", 224))
    model = _build_model(model_name, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, classes, model_name, image_size


def _read_test_samples(
    processed_root: Path,
    classes: list[str],
) -> list[_Sample]:
    """Read the test split and source labels from manifest.csv."""
    manifest_path = processed_root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Test manifest not found: {manifest_path}")
    class_to_index = {name: index for index, name in enumerate(classes)}
    samples: list[_Sample] = []
    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        required = {"path", "class", "split", "source"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Manifest is missing columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            if row["split"].strip() != "test":
                continue
            class_name = row["class"].strip()
            source = row["source"].strip().casefold().replace("_", "-")
            if class_name not in class_to_index:
                raise ValueError(
                    f"Unknown class {class_name!r} in test manifest."
                )
            if source in {"downloaded", "download"}:
                source = "public"
            elif source in {"camera", "own-camera"}:
                source = "own"
            if source not in {"public", "own"}:
                raise ValueError(f"Unknown source {row['source']!r} in manifest.")
            image_path = Path(row["path"])
            if not image_path.is_absolute():
                image_path = ROOT / image_path
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Manifest image not found: {image_path}"
                )
            samples.append(
                _Sample(image_path, class_to_index[class_name], source)
            )
    if not samples:
        raise ValueError(f"No test samples found in {manifest_path}")
    return samples


def _run_inference(
    model: nn.Module,
    loader: DataLoader,
    dataset: _ManifestDataset,
) -> _Predictions:
    """Run CPU inference and return aligned labels, predictions, and metadata."""
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            all_labels.extend(labels.tolist())
            all_preds.extend(probabilities.argmax(axis=1).tolist())
            all_probs.append(probabilities)
    return _Predictions(
        labels=np.asarray(all_labels, dtype=np.int64),
        preds=np.asarray(all_preds, dtype=np.int64),
        probs=np.concatenate(all_probs, axis=0),
        paths=[str(sample.path) for sample in dataset.samples],
        sources=[sample.source for sample in dataset.samples],
    )


def _subset_predictions(
    predictions: _Predictions,
    mask: np.ndarray,
) -> _Predictions:
    """Return a metadata-aligned subset of prediction results."""
    indices = np.flatnonzero(mask)
    return _Predictions(
        labels=predictions.labels[indices],
        preds=predictions.preds[indices],
        probs=predictions.probs[indices],
        paths=[predictions.paths[index] for index in indices],
        sources=[predictions.sources[index] for index in indices],
    )


def _print_report(
    title: str,
    predictions: _Predictions,
    classes: list[str],
) -> float:
    """Print per-class and aggregate metrics, returning macro F1."""
    print(f"\n{title} (n={len(predictions.labels)})")
    if len(predictions.labels) == 0:
        print("  No images in this view.")
        return float("nan")

    report = classification_report(
        predictions.labels,
        predictions.preds,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0,
        output_dict=True,
    )
    print(f"  {'Class':<14} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>9}")
    for class_name in classes:
        row = cast(dict[str, float], report[class_name])
        print(
            f"  {class_name:<14} {row['precision']:>10.4f} "
            f"{row['recall']:>10.4f} {row['f1-score']:>10.4f} "
            f"{int(row['support']):>9}"
        )
    accuracy = float((predictions.labels == predictions.preds).mean())
    macro_f1 = float(
        f1_score(
            predictions.labels,
            predictions.preds,
            labels=list(range(len(classes))),
            average="macro",
            zero_division=0,
        )
    )
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    return macro_f1


def _save_confusion_matrix(
    predictions: _Predictions,
    classes: list[str],
    output_path: Path,
    title: str,
) -> None:
    """Save a row-normalized confusion matrix with raw cell counts."""
    matrix = confusion_matrix(
        predictions.labels,
        predictions.preds,
        labels=list(range(len(classes))),
    )
    normalized = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(
        normalized,
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
        label="Recall (row-normalized)",
    )
    axis.set_xticks(range(len(classes)), labels=classes, rotation=45, ha="right")
    axis.set_yticks(range(len(classes)), labels=classes)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)
    for row in range(len(classes)):
        for column in range(len(classes)):
            color = "white" if normalized[row, column] > 0.6 else "black"
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color=color,
            )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"  Confusion matrix saved: {output_path}")


def _print_empty_counts(
    predictions: _Predictions,
    classes: list[str],
) -> None:
    """Print counts for real items mistaken as empty and empty trays mistaken."""
    empty_index = len(classes) - 1 if classes else None
    if empty_index is None:
        print("\nEmpty-class error counts unavailable: no configured classes.")
        return
    real_to_empty = int(
        np.sum((predictions.labels != empty_index) & (predictions.preds == empty_index))
    )
    empty_to_item = int(
        np.sum((predictions.labels == empty_index) & (predictions.preds != empty_index))
    )
    print("\nEmpty-tray error counts (all test images):")
    print(f"  Real items predicted as empty: {real_to_empty}")
    print(f"  Empty trays predicted as another class: {empty_to_item}")


def _print_top_confusions(
    predictions: _Predictions,
    classes: list[str],
) -> None:
    """Print up to ten most frequent off-diagonal confusion pairs."""
    matrix = confusion_matrix(
        predictions.labels,
        predictions.preds,
        labels=list(range(len(classes))),
    )
    pairs = [
        (int(matrix[true_index, predicted_index]), true_index, predicted_index)
        for true_index in range(len(classes))
        for predicted_index in range(len(classes))
        if true_index != predicted_index and matrix[true_index, predicted_index] > 0
    ]
    pairs.sort(reverse=True)
    print("\nTop 10 confusion pairs (true -> predicted):")
    if not pairs:
        print("  No incorrect predictions.")
        return
    print(f"  {'True':<14} {'Predicted':<14} {'Count':>7}")
    for count, true_index, predicted_index in pairs[:10]:
        print(
            f"  {classes[true_index]:<14} "
            f"{classes[predicted_index]:<14} {count:>7}"
        )


def _save_own_errors(
    predictions: _Predictions,
    classes: list[str],
    output_root: Path,
) -> None:
    """Copy the 40 most confident incorrect own-camera images into class folders."""
    own_indices = [
        index
        for index, source in enumerate(predictions.sources)
        if source == "own" and predictions.labels[index] != predictions.preds[index]
    ]
    own_indices.sort(
        key=lambda index: float(predictions.probs[index].max()),
        reverse=True,
    )
    selected = own_indices[:40]
    for index in selected:
        true_class = classes[int(predictions.labels[index])]
        predicted_class = classes[int(predictions.preds[index])]
        confidence = float(predictions.probs[index].max())
        source_path = Path(predictions.paths[index])
        target_dir = output_root / f"{true_class}__{predicted_class}"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / (
            f"confidence_{confidence:.4f}__{source_path.name}"
        )
        shutil.copy2(source_path, target_path)
    print(
        f"\nSaved {len(selected)} most confident own-camera errors "
        f"under {output_root}"
    )


def _threshold_sweep(
    predictions: _Predictions,
) -> None:
    """Report own-camera coverage and accepted accuracy across confidence cutoffs."""
    own_mask = np.asarray(
        [source == "own" for source in predictions.sources],
        dtype=bool,
    )
    own_predictions = _subset_predictions(predictions, own_mask)
    if not len(own_predictions.labels):
        print("\nThreshold sweep: no own-camera test images.")
        return
    max_confidence = own_predictions.probs.max(axis=1)
    thresholds = np.round(np.arange(0.5, 0.951, 0.05), 2)
    suggested_threshold: float | None = None
    print("\nOwn-camera confidence threshold sweep:")
    print(f"  {'Threshold':>9} {'Accepted':>10} {'Share':>9} {'Accuracy':>11}")
    for threshold in thresholds:
        accepted = max_confidence >= threshold
        count = int(accepted.sum())
        share = count / len(own_predictions.labels)
        accuracy = (
            float(
                (own_predictions.preds[accepted] == own_predictions.labels[accepted])
                .mean()
            )
            if count
            else float("nan")
        )
        print(
            f"  {threshold:>9.2f} {count:>10} {share:>8.1%} "
            f"{accuracy:>11.4f}"
        )
        if count and accuracy >= 0.95 and suggested_threshold is None:
            suggested_threshold = float(threshold)
    if suggested_threshold is None:
        print(
            "  No tested threshold reached 0.95 accepted-image accuracy; "
            "no suggestion."
        )
    else:
        print(
            f"  Suggested threshold (not applied): {suggested_threshold:.2f}"
        )


def _compare_checkpoint(
    compare_path: Path,
    test_samples: list[_Sample],
    current_macro_f1: float,
    expected_classes: list[str],
) -> None:
    """Evaluate an older checkpoint on the identical test images and compare F1."""
    model, classes, model_name, image_size = _load_model(compare_path)
    if classes != expected_classes:
        raise ValueError(
            f"Old checkpoint classes {classes} do not match "
            f"current classes {expected_classes}"
        )
    dataset = _ManifestDataset(test_samples, _val_transform(image_size))
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    old_predictions = _run_inference(model, loader, dataset)
    old_macro_f1 = float(
        f1_score(
            old_predictions.labels,
            old_predictions.preds,
            labels=list(range(len(classes))),
            average="macro",
            zero_division=0,
        )
    )
    print("\nCheckpoint macro F1 comparison (same test split):")
    print(f"  Current {model_name}: {current_macro_f1:.4f}")
    print(f"  Old {compare_path}: {old_macro_f1:.4f}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse evaluation command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        metavar="CHECKPOINT",
        help="Evaluate an older checkpoint on the same test split",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Evaluate all test sources and print metrics, errors, and comparisons."""
    args = parse_args(argv)
    config = load_config()
    configured_classes = [str(label) for label in config["classes"]]
    processed_root = ROOT / config["paths"]["processed_root"]
    models_dir = ROOT / "models"
    checkpoint_path = models_dir / "best.pt"

    print(f"Loading checkpoint: {checkpoint_path}")
    model, classes, model_name, image_size = _load_model(
        checkpoint_path,
        configured_classes,
    )
    samples = _read_test_samples(processed_root, classes)
    dataset = _ManifestDataset(samples, _val_transform(image_size))
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    print(
        f"Model: {model_name} | image size: {image_size} | "
        f"test images: {len(dataset)}"
    )
    predictions = _run_inference(model, loader, dataset)

    public_mask = np.asarray(
        [source == "public" for source in predictions.sources],
        dtype=bool,
    )
    own_mask = np.asarray(
        [source == "own" for source in predictions.sources],
        dtype=bool,
    )
    views = [
        ("All test images", predictions, "confusion_matrix_all.png"),
        (
            "Public-source test images",
            _subset_predictions(predictions, public_mask),
            "confusion_matrix_public.png",
        ),
        (
            "Own-camera test images",
            _subset_predictions(predictions, own_mask),
            "confusion_matrix_own.png",
        ),
    ]
    current_macro_f1 = float("nan")
    for title, view_predictions, filename in views:
        view_f1 = _print_report(title, view_predictions, classes)
        if title == "All test images":
            current_macro_f1 = view_f1
        _save_confusion_matrix(
            view_predictions,
            classes,
            models_dir / filename,
            title,
        )

    _print_empty_counts(predictions, classes)
    _print_top_confusions(predictions, classes)
    _save_own_errors(predictions, classes, models_dir / "errors")
    _threshold_sweep(predictions)
    if args.compare is not None:
        compare_path = args.compare
        if not compare_path.is_absolute():
            compare_path = ROOT / compare_path
        _compare_checkpoint(
            compare_path,
            samples,
            current_macro_f1,
            configured_classes,
        )


if __name__ == "__main__":
    main()
