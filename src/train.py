"""Fine-tune a pretrained image classifier for waste sorting."""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageEnhance
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import get_classes, load_config, set_seed  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EARLY_STOP_PATIENCE = 5
DEFAULT_SOURCE_WEIGHTS = {"public": 1.0, "own": 1.0}


class RandomGamma:
    """Apply a randomly selected gamma adjustment to a PIL image."""

    def __init__(self, gamma_range: tuple[float, float]) -> None:
        """Store the inclusive lower and upper gamma values."""
        self.gamma_range = gamma_range

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply a random gamma adjustment to ``image``."""
        gamma = random.uniform(*self.gamma_range)
        return TF.adjust_gamma(image, gamma)


class RandomSoftShadow:
    """Darken a random polygon on an image to simulate a soft shadow."""

    def __init__(self, probability: float) -> None:
        """Store the probability of adding a shadow."""
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        """Return a copy of ``image`` with an optional darkened polygon."""
        if random.random() >= self.probability:
            return image

        width, height = image.size
        mask = Image.new("L", image.size, 0)
        points = [
            (
                random.randint(0, width),
                random.randint(0, height),
            )
            for _ in range(random.randint(4, 7))
        ]
        ImageDraw.Draw(mask).polygon(points, fill=255)
        darkened = ImageEnhance.Brightness(image).enhance(random.uniform(0.6, 0.9))
        return Image.composite(darkened, image, mask)


class AddGaussianNoise:
    """Add mild zero-mean Gaussian noise to a tensor image."""

    def __init__(self, standard_deviation: float) -> None:
        """Store the standard deviation for the generated noise."""
        self.standard_deviation = standard_deviation

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Return ``image`` with mild noise, clipped to the image range."""
        noise = torch.randn_like(image) * self.standard_deviation
        return (image + noise).clamp(0.0, 1.0)


def make_train_transform(
    image_size: int = 224,
    is_empty: bool = False,
) -> transforms.Compose:
    """Build the camera-oriented training augmentation pipeline."""
    brightness = 0.5 if is_empty else 0.3
    shadow_probability = 0.65 if is_empty else 0.3
    gamma_range = (0.55, 1.5) if is_empty else (0.8, 1.2)
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=brightness,
                contrast=0.3,
                saturation=0.3,
                hue=0.04,
            ),
            RandomGamma(gamma_range),
            RandomSoftShadow(shadow_probability),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
                p=0.2,
            ),
            transforms.ToTensor(),
            transforms.RandomApply([AddGaussianNoise(0.02)], p=0.2),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        ]
    )


def make_val_transform(image_size: int = 224) -> transforms.Compose:
    """Build the deterministic validation image transform."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class ManifestImageDataset(Dataset[tuple[torch.Tensor, int]]):
    """Load split samples and retain their manifest source information."""

    def __init__(
        self,
        records: list[dict[str, str]],
        classes: list[str],
        transform: transforms.Compose,
        empty_transform: transforms.Compose | None = None,
    ) -> None:
        """Create a dataset from manifest rows and ordered class names."""
        self.classes = list(classes)
        self.class_to_idx = {name: index for index, name in enumerate(classes)}
        self.transform = transform
        self.empty_transform = empty_transform or transform
        self.samples: list[tuple[str, int]] = []
        self.sources: list[str] = []
        for record in records:
            class_name = record["class"]
            if class_name not in self.class_to_idx:
                raise ValueError(f"Unknown class in manifest: {class_name!r}")
            self.samples.append(
                (record["image_path"], self.class_to_idx[class_name])
            )
            self.sources.append(record["source"])
        self.targets = [target for _, target in self.samples]

    def __len__(self) -> int:
        """Return the number of manifest samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Load and transform one sample and return its class index."""
        image_path, target = self.samples[index]
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        transform = (
            self.empty_transform
            if target == len(self.classes) - 1
            else self.transform
        )
        return transform(image), target


def normalize_source(source: str) -> str:
    """Map manifest source spellings to the supported source-weight keys."""
    normalized = source.strip().casefold().replace("_", "-")
    if normalized in {"public", "downloaded", "download"}:
        return "public"
    if normalized in {"own", "own-camera", "camera"}:
        return "own"
    raise ValueError(f"Unsupported manifest source: {source!r}")


def read_manifest_records(processed_root: Path) -> tuple[
    list[dict[str, str]], list[dict[str, str]]
]:
    """Read train and validation rows from the processed manifest."""
    manifest_path = processed_root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Training manifest not found: {manifest_path}")

    train_records: list[dict[str, str]] = []
    val_records: list[dict[str, str]] = []
    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        required_columns = {"path", "class", "split", "source"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Manifest is missing columns: {', '.join(sorted(missing_columns))}"
            )
        for row in reader:
            if row["split"] not in {"train", "val"}:
                continue
            raw_path = Path(row["path"])
            image_path = raw_path if raw_path.is_absolute() else ROOT / raw_path
            if not image_path.is_file():
                raise FileNotFoundError(f"Manifest image not found: {image_path}")
            record = {
                "image_path": str(image_path),
                "class": row["class"].strip(),
                "source": normalize_source(row["source"]),
            }
            if row["split"] == "train":
                train_records.append(record)
            else:
                val_records.append(record)
    if not train_records or not val_records:
        raise ValueError("Manifest must contain both train and val samples.")
    return train_records, val_records


def load_datasets(
    processed_root: Path,
    batch_size: int,
    num_workers: int,
    source_weights: dict[str, float] | None = None,
    seed: int = 42,
    image_size: int = 224,
) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    """Return manifest-backed loaders and inverse-frequency class weights."""
    classes = get_classes()
    train_records, val_records = read_manifest_records(processed_root)
    for split_name, records in (("train", train_records), ("val", val_records)):
        found = {record["class"] for record in records}
        missing = set(classes) - found
        extra = found - set(classes)
        if missing or extra:
            raise ValueError(
                f"{split_name} manifest class mismatch — "
                f"missing: {sorted(missing)}, extra: {sorted(extra)}"
            )

    train_ds = ManifestImageDataset(
        train_records,
        classes,
        make_train_transform(image_size),
        make_train_transform(image_size, is_empty=True),
    )
    val_ds = ManifestImageDataset(
        val_records,
        classes,
        make_val_transform(image_size),
    )

    class_counts = torch.bincount(
        torch.tensor(train_ds.targets, dtype=torch.long),
        minlength=len(classes),
    ).to(dtype=torch.float32)
    class_weights = class_counts.sum() / (
        len(classes) * class_counts.clamp(min=1)
    )

    effective_source_weights = source_weights or DEFAULT_SOURCE_WEIGHTS
    sample_weights = [
        float(class_weights[target]) * effective_source_weights[source]
        for target, source in zip(train_ds.targets, train_ds.sources)
    ]
    sampler_generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(train_ds),
        replacement=True,
        generator=sampler_generator,
    )
    loader_generator = torch.Generator().manual_seed(seed)
    persistent_workers = sys.platform.startswith("linux") and num_workers > 0
    loader_options: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        loader_options["prefetch_factor"] = 2
        loader_options["generator"] = loader_generator
    train_loader = DataLoader(train_ds, sampler=sampler, **loader_options)
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **loader_options,
    )
    return train_loader, val_loader, class_weights


def build_model(arch: str, num_classes: int) -> nn.Module:
    """Return a pretrained model with a replacement classifier head."""
    model_builders: dict[str, tuple[Any, Any]] = {
        "efficientnet_b0": (
            models.efficientnet_b0,
            models.EfficientNet_B0_Weights.DEFAULT,
        ),
        "mobilenet_v3_large": (
            models.mobilenet_v3_large,
            models.MobileNet_V3_Large_Weights.DEFAULT,
        ),
        "mobilenet_v3_small": (
            models.mobilenet_v3_small,
            models.MobileNet_V3_Small_Weights.DEFAULT,
        ),
        "convnext_tiny": (
            models.convnext_tiny,
            models.ConvNeXt_Tiny_Weights.DEFAULT,
        ),
    }
    if arch not in model_builders:
        raise ValueError(f"Unsupported architecture: {arch!r}")
    builder, weights = model_builders[arch]
    model = builder(weights=weights)
    last_layer = model.classifier[-1]
    if not isinstance(last_layer, nn.Linear):
        raise TypeError(f"Unexpected classifier structure for {arch!r}")
    model.classifier[-1] = nn.Linear(last_layer.in_features, num_classes)
    return model


def freeze_backbone(model: nn.Module, arch: str) -> None:
    """Freeze the feature backbone and leave the classifier trainable."""
    del arch
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def unfreeze_last_blocks(model: nn.Module, arch: str) -> None:
    """Unfreeze the full network for the second training stage."""
    del arch
    for parameter in model.parameters():
        parameter.requires_grad = True


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a short linear warmup followed by cosine decay."""
    step_count = max(1, total_steps)
    warmup_steps = max(1, int(step_count * 0.05))

    def learning_rate_scale(step: int) -> float:
        """Return the warmup and cosine multiplier for an optimizer step."""
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        decay_steps = step_count - warmup_steps
        if decay_steps <= 0:
            return 1.0
        progress = min(1.0, (step - warmup_steps + 1) / decay_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    max_batches: int | None,
    scaler: torch.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    use_amp: bool = False,
) -> tuple[float, list[int], list[int]]:
    """Run one pass and return average loss, predictions, and labels."""
    is_train = optimizer is not None
    model.train(is_train)
    if is_train and not any(
        parameter.requires_grad for parameter in model.features.parameters()
    ):
        model.features.eval()

    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    grad_context = torch.enable_grad() if is_train else torch.no_grad()
    with grad_context:
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device, non_blocking=use_amp)
            if device.type == "cuda":
                images = images.to(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=use_amp)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, labels)
            optimizer_updated = True
            if is_train:
                if scaler is not None:
                    old_scale = scaler.get_scale()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_updated = scaler.get_scale() >= old_scale
                else:
                    loss.backward()
                    optimizer.step()
            if scheduler is not None and optimizer_updated:
                scheduler.step()

            total_loss += loss.item() * images.size(0)
            all_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())
    sample_count = len(all_labels)
    if sample_count == 0:
        raise ValueError("No batches were processed; check --max-batches and data.")
    return total_loss / sample_count, all_preds, all_labels


def compute_metrics(
    preds: list[int],
    labels: list[int],
    num_classes: int,
) -> tuple[float, float]:
    """Return accuracy and macro F1 for the requested class indices."""
    if not labels:
        return 0.0, 0.0
    correct = sum(prediction == label for prediction, label in zip(preds, labels))
    accuracy = correct / len(labels)
    macro_f1 = f1_score(
        labels,
        preds,
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0,
    )
    return accuracy, float(macro_f1)


def compute_class_recalls(
    preds: list[int],
    labels: list[int],
    num_classes: int,
) -> list[float]:
    """Return per-class recall, using zero for classes absent from the source."""
    recalls: list[float] = []
    for class_index in range(num_classes):
        class_labels = [
            prediction
            for prediction, label in zip(preds, labels)
            if label == class_index
        ]
        correct = sum(prediction == class_index for prediction in class_labels)
        recalls.append(correct / len(class_labels) if class_labels else 0.0)
    return recalls


def parse_source_weights(value: str) -> dict[str, float]:
    """Parse comma-separated public and own source multipliers."""
    parsed: dict[str, float] = {}
    try:
        for part in value.split(","):
            key, raw_weight = part.split("=", maxsplit=1)
            key = key.strip()
            weight = float(raw_weight)
            if key not in DEFAULT_SOURCE_WEIGHTS or key in parsed:
                raise ValueError
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError
            parsed[key] = weight
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected positive weights as public=1.0,own=1.0"
        ) from exc
    if set(parsed) != set(DEFAULT_SOURCE_WEIGHTS):
        raise argparse.ArgumentTypeError(
            "both public and own weights are required, e.g. public=1.0,own=1.0"
        )
    return parsed


def resolve_device(device_option: str) -> torch.device:
    """Select the requested device and print available GPU information."""
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        total_vram_gib = properties.total_memory / (1024**3)
        print(
            f"GPU: {properties.name} | total VRAM: {total_vram_gib:.2f} GiB",
            flush=True,
        )
    else:
        print("GPU: unavailable | total VRAM: unavailable", flush=True)

    if device_option == "cuda" and not cuda_available:
        raise RuntimeError(
            "--device cuda was requested, but torch.cuda.is_available() is False."
        )
    if device_option == "auto" and not cuda_available:
        print(
            "Warning: CUDA is unavailable; training will fall back to CPU.",
            flush=True,
        )
    selected = "cuda" if device_option == "auto" and cuda_available else device_option
    if selected == "auto":
        selected = "cpu"
    print(f"Training device: {selected}", flush=True)
    return torch.device(selected)


def run_training(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    device: torch.device,
    batch_size: int,
    seed: int,
) -> None:
    """Run both fine-tuning stages and write the best checkpoint and log."""
    set_seed(seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    processed_root = ROOT / cfg["paths"]["processed_root"]
    labels_path = ROOT / cfg["paths"]["labels"]
    models_dir = labels_path.parent
    models_dir.mkdir(parents=True, exist_ok=True)
    image_size = int(cfg.get("image_size", 224))
    classes = get_classes()

    print(f"Loading datasets from {processed_root}", flush=True)
    train_loader, val_loader, class_weights = load_datasets(
        processed_root,
        batch_size=batch_size,
        num_workers=args.num_workers,
        source_weights=args.source_weights,
        seed=seed,
        image_size=image_size,
    )
    print(
        f"  train: {len(train_loader.dataset)} images | "
        f"val: {len(val_loader.dataset)} images | classes: {classes}",
        flush=True,
    )

    print(f"Building model: {args.model}", flush=True)
    model = build_model(args.model, len(classes)).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=0.1,
    )

    best_checkpoint = models_dir / "best.pt"
    log_path = models_dir / "training_log.csv"
    log_fields = [
        "stage",
        "epoch",
        "train_loss",
        "val_accuracy",
        "val_macro_f1",
        "public_val_macro_f1",
        "own_val_macro_f1",
        "own_val_recall",
        "elapsed_seconds",
        "peak_gpu_memory_gib",
    ]
    log_fields.extend(f"own_recall_{class_name}" for class_name in classes)
    best_f1 = -1.0
    no_improve = 0
    global_epoch = 0

    def run_stage(
        stage: int,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        csv_writer: csv.DictWriter,
    ) -> bool:
        """Train one stage and report whether patience stopped it early."""
        nonlocal best_f1, no_improve, global_epoch
        batches_per_epoch = len(train_loader)
        if args.max_batches is not None:
            batches_per_epoch = min(batches_per_epoch, args.max_batches)
        scheduler = make_scheduler(optimizer, epochs * batches_per_epoch)

        for stage_epoch in range(1, epochs + 1):
            global_epoch += 1
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started_at = time.perf_counter()
            train_loss, _, _ = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                args.max_batches,
                scaler=scaler,
                scheduler=scheduler,
                use_amp=use_amp,
            )
            _, val_preds, val_labels = run_epoch(
                model,
                val_loader,
                criterion,
                None,
                device,
                args.max_batches,
                use_amp=use_amp,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak_memory_gib = (
                    torch.cuda.max_memory_allocated(device) / (1024**3)
                )
            else:
                peak_memory_gib = 0.0
            elapsed = time.perf_counter() - started_at
            val_accuracy, val_macro_f1 = compute_metrics(
                val_preds, val_labels, len(classes)
            )
            val_sources = val_loader.dataset.sources[: len(val_labels)]
            source_f1: dict[str, float] = {}
            source_recalls: dict[str, list[float]] = {}
            for source in ("public", "own"):
                source_predictions = [
                    prediction
                    for prediction, sample_source in zip(val_preds, val_sources)
                    if sample_source == source
                ]
                source_labels = [
                    label
                    for label, sample_source in zip(val_labels, val_sources)
                    if sample_source == source
                ]
                _, source_f1[source] = compute_metrics(
                    source_predictions, source_labels, len(classes)
                )
                source_recalls[source] = compute_class_recalls(
                    source_predictions, source_labels, len(classes)
                )

            own_recall_text = ", ".join(
                f"{class_name}={recall:.3f}"
                for class_name, recall in zip(classes, source_recalls["own"])
            )
            print(
                f"{stage:>5} {stage_epoch:>5}/{epochs:<5} "
                f"{train_loss:>10.4f} {val_accuracy:>10.4f} "
                f"{val_macro_f1:>10.4f} {source_f1['public']:>10.4f} "
                f"{source_f1['own']:>10.4f} {elapsed:>9.1f}s "
                f"{peak_memory_gib:>8.2f} GiB | own recall: {own_recall_text}",
                flush=True,
            )
            log_row: dict[str, Any] = {
                "stage": stage,
                "epoch": global_epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_accuracy": f"{val_accuracy:.6f}",
                "val_macro_f1": f"{val_macro_f1:.6f}",
                "public_val_macro_f1": f"{source_f1['public']:.6f}",
                "own_val_macro_f1": f"{source_f1['own']:.6f}",
                "own_val_recall": own_recall_text,
                "elapsed_seconds": f"{elapsed:.2f}",
                "peak_gpu_memory_gib": f"{peak_memory_gib:.3f}",
            }
            log_row.update(
                {
                    f"own_recall_{class_name}": f"{recall:.6f}"
                    for class_name, recall in zip(
                        classes, source_recalls["own"]
                    )
                }
            )
            csv_writer.writerow(log_row)

            if val_macro_f1 > best_f1:
                best_f1 = val_macro_f1
                no_improve = 0
                torch.save(
                    {
                        "epoch": global_epoch,
                        "arch": args.model,
                        "model_name": args.model,
                        "image_size": image_size,
                        "state_dict": {
                            key: value.detach().cpu()
                            for key, value in model.state_dict().items()
                        },
                        "val_f1": val_macro_f1,
                        "val_acc": val_accuracy,
                        "classes": classes,
                    },
                    best_checkpoint,
                )
                print(
                    f"  New best whole-validation macro F1 "
                    f"{best_f1:.4f}; saved {best_checkpoint}",
                    flush=True,
                )
            else:
                no_improve += 1
                if no_improve >= EARLY_STOP_PATIENCE:
                    print(
                        f"  Early stopping: no validation macro F1 improvement "
                        f"for {EARLY_STOP_PATIENCE} epochs.",
                        flush=True,
                    )
                    return True
        return False

    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        csv_writer = csv.DictWriter(log_file, fieldnames=log_fields)
        csv_writer.writeheader()
        print(
            "\nStage Epoch   TrainLoss  ValAccuracy    ValF1  "
            "PublicF1     OwnF1  Seconds PeakVRAM | Own-camera per-class recall"
        )
        freeze_backbone(model, args.model)
        stage1_optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=1e-3,
            weight_decay=1e-4,
        )
        print(f"\nStage 1: frozen backbone for {args.stage1_epochs} epoch(s)")
        stopped = run_stage(
            1,
            args.stage1_epochs,
            stage1_optimizer,
            csv_writer,
        )

        if not stopped and args.stage2_epochs > 0:
            unfreeze_last_blocks(model, args.model)
            stage2_optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-4,
                weight_decay=1e-4,
            )
            print(f"\nStage 2: full network unfrozen for {args.stage2_epochs} epoch(s)")
            run_stage(2, args.stage2_epochs, stage2_optimizer, csv_writer)
        log_file.flush()

    if best_f1 < 0:
        raise RuntimeError("Training finished without a validation checkpoint.")
    print(
        f"\nTraining complete. Best whole-validation macro F1={best_f1:.4f}",
        flush=True,
    )
    print(f"Checkpoint: {best_checkpoint}", flush=True)
    print(f"Training log: {log_path}", flush=True)


def is_cuda_out_of_memory(error: BaseException) -> bool:
    """Return whether ``error`` represents a CUDA allocation failure."""
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "CUDA out of memory" in str(error)
    )


def train(args: argparse.Namespace) -> None:
    """Select the device, retry CUDA OOM once, and run training."""
    cfg = load_config()
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    device = resolve_device(args.device)
    batch_size = args.batch_size
    try:
        run_training(args, cfg, device, batch_size, seed)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
        if device.type != "cuda" or not is_cuda_out_of_memory(error):
            raise
        if batch_size <= 1:
            raise RuntimeError(
                "CUDA out of memory at batch size 1; no smaller retry is possible."
            ) from error
        retry_batch_size = max(1, batch_size // 2)
        print(
            f"CUDA out of memory at batch size {batch_size}; retrying once "
            f"with batch size {retry_batch_size}.",
            flush=True,
        )
        torch.cuda.empty_cache()
        try:
            run_training(args, cfg, device, retry_batch_size, seed)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as retry_error:
            if is_cuda_out_of_memory(retry_error):
                raise RuntimeError(
                    f"CUDA out of memory again at batch size {retry_batch_size}; "
                    "the one permitted retry failed."
                ) from retry_error
            raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the training command-line options."""
    default_workers = 4 if sys.platform.startswith("linux") else 2
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=[
            "efficientnet_b0",
            "mobilenet_v3_large",
            "mobilenet_v3_small",
            "convnext_tiny",
        ],
        default="efficientnet_b0",
        help="Pretrained backbone architecture",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Training device",
    )
    parser.add_argument(
        "--stage1-epochs",
        type=int,
        default=3,
        dest="stage1_epochs",
        help="Epochs with frozen backbone",
    )
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=20,
        dest="stage2_epochs",
        help="Epochs with the full network unfrozen",
    )
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=default_workers,
        dest="num_workers",
        help="DataLoader worker processes",
    )
    parser.add_argument(
        "--source-weights",
        type=parse_source_weights,
        default=DEFAULT_SOURCE_WEIGHTS.copy(),
        help="Relative per-image source multipliers",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        dest="max_batches",
        metavar="N",
        help="Limit batches per epoch (smoke-test mode)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args(argv)
    if args.stage1_epochs < 0 or args.stage2_epochs < 0:
        parser.error("epoch counts must be non-negative")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be at least 1")
    return args


if __name__ == "__main__":
    train(parse_args())
