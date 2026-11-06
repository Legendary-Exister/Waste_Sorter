"""Copy raw and own waste photos into resized train, val and test image folders."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

from PIL import Image, UnidentifiedImageError

if __package__:
    from .common import ROOT, get_classes, load_config, set_seed
else:
    from common import ROOT, get_classes, load_config, set_seed


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
HF_DATASET = "TrashBusters/combined"
SPLITS = ("train", "val", "test")
PHASH_THRESHOLD = 4
MIN_OWN_IMAGES = 30


@dataclass(frozen=True)
class Sample:
    """A source image together with the class it maps onto."""

    dataset: str
    target: str
    path: Path
    item_id: str | None
    source: str = "public"


def normalize_name(name: str) -> str:
    """Normalize a source label for matching configured aliases."""
    return re.sub(r"[ _-]+", " ", name.strip().lower())


def load_class_map(path: Path) -> dict[str, str]:
    """Load source-folder -> target-class aliases from class_map.yaml."""
    try:
        from ruamel.yaml import YAML
        from ruamel.yaml.error import YAMLError as YAMLError
    except ImportError:
        import yaml
        from yaml import YAMLError as YAMLError

        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, YAMLError) as exc:
            raise RuntimeError(f"Could not load class map {path}: {exc}") from exc
    else:
        try:
            with path.open(encoding="utf-8") as handle:
                raw = YAML(typ="safe").load(handle) or {}
        except (OSError, YAMLError) as exc:
            raise RuntimeError(f"Could not load class map {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"class map {path} must be a mapping of names to classes")
    return {normalize_name(str(key)): str(value) for key, value in raw.items()}


def resolve_class(name: str, class_map: dict[str, str], classes: list[str]) -> str | None:
    """Return the configured output class for a source folder name, if mapped."""
    value = class_map.get(normalize_name(name))
    return value if value in classes else None


def walk_images(folder: Path) -> Iterator[Path]:
    """Yield supported image files under a folder in a stable order."""
    if not folder.is_dir():
        return
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def count_images(folder: Path) -> int:
    """Count supported image files under a folder."""
    return sum(1 for _ in walk_images(folder))


def load_yolo_class_maps(
    raw_root: Path, class_map: dict[str, str], classes: list[str]
) -> dict[Path, dict[int, str | None]]:
    """Load Roboflow data.yaml class ids and map their names to project classes."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Reading Roboflow data.yaml files requires PyYAML.") from exc

    mappings: dict[Path, dict[int, str | None]] = {}
    for metadata_path in sorted(raw_root.rglob("data.yaml")) if raw_root.is_dir() else []:
        try:
            config = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise RuntimeError(f"Could not load Roboflow metadata {metadata_path}: {exc}") from exc
        names = config.get("names")
        if isinstance(names, list):
            entries = enumerate(names)
        elif isinstance(names, dict):
            entries = ((int(key), value) for key, value in names.items())
        else:
            raise ValueError(f"{metadata_path} must define class names as a list or mapping")
        class_ids: dict[int, str | None] = {}
        for class_id, name in entries:
            normalized = normalize_name(str(name))
            if normalized == "objects":
                class_ids[int(class_id)] = "other"
            else:
                class_ids[int(class_id)] = resolve_class(str(name), class_map, classes)
        mappings[metadata_path.parent.resolve()] = class_ids
    return mappings


def _yolo_target(
    image: Path, yolo_maps: dict[Path, dict[int, str | None]]
) -> tuple[str | None, str | None]:
    """Map a detection image using the class of its largest annotated box."""
    dataset_root = next((parent.resolve() for parent in image.parents if parent.resolve() in yolo_maps), None)
    if dataset_root is None:
        return None, None
    images_dir = next((parent for parent in image.parents if parent.name.lower() == "images"), None)
    if images_dir is None:
        return None, "image is not in an images directory"
    annotation = images_dir.parent / "labels" / f"{image.stem}.txt"
    if not annotation.is_file():
        return None, f"missing annotation {annotation.relative_to(ROOT)}"
    try:
        lines = annotation.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return None, f"cannot read annotation {annotation.relative_to(ROOT)}: {exc}"
    if not lines or not any(line.strip() for line in lines):
        return None, "empty annotation"

    largest_area = -1.0
    selected: str | None = None
    for line in lines:
        fields = line.split()
        if len(fields) < 5:
            return None, f"invalid YOLO annotation in {annotation.relative_to(ROOT)}"
        try:
            class_id = int(fields[0])
            area = float(fields[3]) * float(fields[4])
        except ValueError:
            return None, f"invalid YOLO annotation in {annotation.relative_to(ROOT)}"
        target = yolo_maps[dataset_root].get(class_id)
        if target is None:
            return None, f"unmapped YOLO class id {class_id} in {annotation.relative_to(ROOT)}"
        if area > largest_area:
            largest_area = area
            selected = target
    return selected, None


def item_id_for(filename: str) -> str:
    """Return the itemid from an own photo named <itemid>_<n>.jpg."""
    stem = Path(filename).stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def unpack_trashnet(cfg: dict[str, Any]) -> Path | None:
    """Unpack data/trashnet/dataset-resized.zip into data/raw/trashnet/<label>/."""
    paths = cfg["paths"]
    archive = ROOT / str(paths["trashnet_archive"])
    target_root = ROOT / str(paths["raw_root"]) / "trashnet"
    if not archive.is_file():
        print(f"note: no TrashNet archive at {archive}; skipping unpack")
        return None
    existing = count_images(target_root)
    if existing > 0:
        print(f"note: using {existing} existing TrashNet images at {target_root}")
        return target_root

    print(f"unpacking {archive} into {target_root} ...")
    written = 0
    with zipfile.ZipFile(archive) as archive_file:
        for member in archive_file.namelist():
            path = Path(member)
            if path.name.startswith(".") or "__MACOSX" in path.parts:
                continue
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            # Prefer dataset-resized/<label>/<file>; otherwise <label>/<file>.
            parts = path.parts
            if len(parts) >= 3 and parts[0] == "dataset-resized":
                label, filename = parts[1], parts[-1]
            elif len(parts) >= 2:
                label, filename = parts[-2], parts[-1]
            else:
                continue
            destination = target_root / label / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive_file.open(member) as source, destination.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            written += 1
    print(f"wrote {written} TrashNet images to {target_root}")
    return target_root


def materialize_combined_parquet(cfg: dict[str, Any]) -> Path | None:
    """Write local combined parquet shards into data/raw/combined/<label>/ as JPEGs."""
    paths = cfg["paths"]
    combined_root = ROOT / str(paths["combined_root"])
    target_root = ROOT / str(paths["raw_root"]) / "combined"
    shards = sorted(combined_root.glob("*.parquet")) if combined_root.is_dir() else []
    if not shards:
        print(f"note: no combined parquet shards at {combined_root}; skipping")
        return None
    existing = count_images(target_root)
    if existing > 0:
        print(f"note: using {existing} existing combined images at {target_root}")
        return target_root

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Reading local combined parquet needs `pyarrow`; install it with `pip install pyarrow`."
        ) from exc

    print(f"materializing {len(shards)} parquet shard(s) into {target_root} ...")
    written = 0
    corrupt = 0
    for shard in shards:
        parquet_file = pq.ParquetFile(shard)
        row_index = 0
        for batch in parquet_file.iter_batches(columns=["image", "label_name"], batch_size=64):
            images = batch.column("image").to_pylist()
            labels = batch.column("label_name").to_pylist()
            for image_payload, label in zip(images, labels):
                destination = target_root / str(label) / f"{shard.stem}_{row_index:06d}.jpg"
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    payload = image_payload["bytes"] if isinstance(image_payload, dict) else None
                    if not payload:
                        raise ValueError("missing image bytes")
                    with Image.open(BytesIO(payload)) as image:
                        image.convert("RGB").save(destination, format="JPEG", quality=95)
                    written += 1
                except (UnidentifiedImageError, OSError, TypeError, ValueError, KeyError) as exc:
                    print(f"WARNING: dropping parquet row {shard.name}#{row_index}: {exc}")
                    corrupt += 1
                row_index += 1
        print(f"  {shard.name}: {row_index} row(s)")
    print(f"wrote {written} combined images to {target_root} (dropped {corrupt} corrupt)")
    return target_root


def collect_samples(
    cfg: dict[str, Any], class_map: dict[str, str], classes: list[str]
) -> tuple[list[Sample], list[str]]:
    """Gather public images and own-camera images from configured/local photo roots."""
    paths = cfg["paths"]
    samples: list[Sample] = []
    skipped: list[str] = []

    roots: list[tuple[Path, str, str | None]] = [
        (ROOT / str(paths["raw_root"]), "public", None),
        (ROOT / str(paths["own_root"]), "own", None),
        (ROOT / "data" / "My pics", "own", None),
        (ROOT / "data" / "Empty tray pics", "own", "empty"),
    ]
    yolo_maps = load_yolo_class_maps(roots[0][0], class_map, classes)
    for source_root, source, forced_target in roots:
        if not source_root.is_dir():
            continue
        for image in walk_images(source_root):
            target = forced_target
            if target is None:
                for parent in image.parents:
                    if parent == source_root:
                        break
                    target = resolve_class(parent.name, class_map, classes)
                    if target is not None:
                        break
            if target is None:
                if source == "public":
                    target, reason = _yolo_target(image, yolo_maps)
                    if reason is not None:
                        skipped.append(f"{image.relative_to(ROOT)} ({reason})")
                else:
                    reason = "no confirmed class mapping"
                if target is None:
                    if reason is None:
                        skipped.append(str(image.parent.relative_to(ROOT)))
                    continue
            dataset = image.relative_to(source_root).parts[0]
            item_id = item_id_for(image.name) if source == "own" else None
            samples.append(Sample(dataset, target, image, item_id, source))

    if not (ROOT / str(paths["raw_root"])).is_dir():
        print(f"note: no raw datasets at {ROOT / str(paths['raw_root'])}")
    if not (ROOT / str(paths["own_root"])).is_dir():
        print(f"note: no own photos at {ROOT / str(paths['own_root'])}")

    return samples, skipped


def _source_roots(cfg: dict[str, Any]) -> list[tuple[Path, str, str | None]]:
    return [
        (ROOT / str(cfg["paths"]["raw_root"]), "public", None),
        (ROOT / str(cfg["paths"]["own_root"]), "own", None),
        (ROOT / "data" / "My pics", "own", None),
        (ROOT / "data" / "Empty tray pics", "own", "empty"),
    ]


def print_inventory(
    cfg: dict[str, Any], class_map: dict[str, str], classes: list[str]
) -> None:
    """Print mapped label folders, direct unmapped image folders, and image counts."""
    rows: dict[tuple[str, str, str], int] = {}
    unmapped: Counter[tuple[str, str]] = Counter()
    roots = _source_roots(cfg)
    yolo_maps = load_yolo_class_maps(roots[0][0], class_map, classes)

    for source_root, source, forced_target in roots:
        if not source_root.is_dir():
            continue
        for image in walk_images(source_root):
            target = forced_target
            label_folder: Path | None = source_root if forced_target else None
            if target is None:
                for parent in image.parents:
                    if parent == source_root:
                        break
                    target = resolve_class(parent.name, class_map, classes)
                    if target is not None:
                        label_folder = parent
                        break
            if target is None or label_folder is None:
                if source == "public":
                    target, reason = _yolo_target(image, yolo_maps)
                    if target is not None:
                        folder = str(image.parent.relative_to(ROOT))
                        key = (source, folder, "YOLO labels -> class_map.yaml (objects -> other)")
                        rows[key] = rows.get(key, 0) + 1
                        continue
                    status = "EMPTY YOLO LABEL; excluded" if reason == "empty annotation" else "UNMAPPED; excluded"
                    unmapped[(source, f"{image.parent.relative_to(ROOT)} [{status}]")] += 1
                else:
                    unmapped[(source, str(image.parent.relative_to(ROOT)))] += 1
            else:
                mapping = (
                    "confirmed source override -> empty"
                    if forced_target is not None
                    else f"class_map.yaml -> {target}"
                )
                key = (source, str(label_folder.relative_to(ROOT)), mapping)
                rows[key] = rows.get(key, 0) + 1

    for source_root, source, forced_target in roots:
        if not source_root.is_dir():
            continue
        for folder in sorted(path for path in source_root.rglob("*") if path.is_dir()):
            target = resolve_class(folder.name, class_map, classes)
            if target is None:
                continue
            key = (source, str(folder.relative_to(ROOT)), f"class_map.yaml -> {target}")
            rows.setdefault(key, 0)

    print("\nImage inventory (source folder counts; class-map status):")
    print(f"{'source':<8} {'images':>8} {'mapping':<48} folder")
    print("-" * 110)
    if not rows:
        print("(no mapped label folders found)")
    for (source, folder, mapping), count in sorted(rows.items()):
        print(f"{source:<8} {count:>8} {mapping:<48} {folder}")
    for (source, folder), count in sorted(unmapped.items()):
        print(f"{source:<8} {count:>8} {'UNMAPPED; not used':<24} {folder}")
    for source_root, source, _ in roots:
        if not source_root.is_dir():
            print(f"{source:<8} {0:>8} {'MISSING SOURCE ROOT':<24} {source_root.relative_to(ROOT)}")
    if unmapped:
        print(
            "\nUnmapped or empty-label images are excluded. Add an explicit class mapping "
            "and confirm it before including any newly unmapped class."
        )


def sample_key(sample: Sample) -> str:
    """Return the grouping key that decides a sample's split.

    Own photos group by itemid, so every picture of one physical item lands in a
    single split. Dataset images group by their own path.
    """
    if sample.source == "own" and sample.item_id is not None:
        return f"own:{sample.item_id}"
    return f"{sample.dataset}:{sample.path.as_posix()}"


def perceptual_hash(
    path: Path | bytes, max_side: int | None = None, quality: int = 90
) -> int:
    """Return a 64-bit difference hash, optionally matching the prepared JPEG pixels."""
    source = BytesIO(path) if isinstance(path, bytes) else path
    with Image.open(source) as image:
        image.load()
        if image.mode in ("P", "PA"):
            image = image.convert("RGBA").convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")
        width, height = image.size
        longest = max(width, height)
        if max_side is not None and longest > max_side:
            scale = max_side / longest
            image = image.resize(
                (max(1, round(width * scale)), max(1, round(height * scale))),
                Image.Resampling.LANCZOS,
            )
        if max_side is not None:
            encoded = BytesIO()
            image.save(encoded, format="JPEG", quality=quality, optimize=True)
            image = Image.open(BytesIO(encoded.getvalue()))
        pixels = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).tobytes()
    value = 0
    for row in range(8):
        offset = row * 9
        for col in range(8):
            value = (value << 1) | int(pixels[offset + col] > pixels[offset + col + 1])
    return value


def hamming_distance(first: int, second: int) -> int:
    """Return the number of differing bits between two perceptual hashes."""
    return (first ^ second).bit_count()


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while value != self.parent[value]:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        a, b = self.find(first), self.find(second)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


class _HashTree:
    """A BK-tree for exact Hamming-radius lookups."""

    def __init__(self) -> None:
        self.root: tuple[int, list[tuple[int, Any]], dict[int, Any]] | None = None

    def add(self, value: int, payload: Any, retain_same_hash: bool = True) -> None:
        if self.root is None:
            self.root = (value, [payload], {})
            return
        node = self.root
        while True:
            distance = hamming_distance(value, node[0])
            if distance == 0:
                if retain_same_hash:
                    node[1].append(payload)
                return
            child = node[2].get(distance)
            if child is None:
                node[2][distance] = (value, [payload], {})
                return
            node = child

    def within(self, value: int, radius: int) -> Iterator[Any]:
        if self.root is None:
            return
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = hamming_distance(value, node[0])
            if distance <= radius:
                yield from node[1]
            low, high = distance - radius, distance + radius
            stack.extend(child for edge, child in node[2].items() if low <= edge <= high)


def _hash_band_specs(threshold: int) -> list[tuple[int, int, int]]:
    count = threshold + 1
    widths = [64 // count + (1 if band < 64 % count else 0) for band in range(count)]
    bands: list[tuple[int, int, int]] = []
    offset = 0
    for band, width in enumerate(widths):
        bands.append((band, offset, (1 << width) - 1))
        offset += width
    return bands


def _near_candidates(
    value: int,
    bands: list[tuple[int, int, int]],
    buckets: dict[tuple[int, int], list[int]],
) -> set[int]:
    candidates: set[int] = set()
    for band, shift, mask in bands:
        candidates.update(buckets[(band, (value >> shift) & mask)])
    return candidates


def _hash_components(
    samples: list[Sample], hashes: list[int], threshold: int
) -> tuple[_UnionFind, _UnionFind]:
    groups = _UnionFind(len(samples))
    near_duplicates = _UnionFind(len(samples))
    own_groups: dict[str, int] = {}
    bands = _hash_band_specs(threshold)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    representatives: dict[int, int] = {}

    for index, (sample, value) in enumerate(zip(samples, hashes)):
        if sample.source == "own":
            key = sample_key(sample)
            previous = own_groups.setdefault(key, index)
            groups.union(index, previous)
        exact = representatives.get(value)
        if exact is not None:
            groups.union(index, exact)
            near_duplicates.union(index, exact)
            continue

        candidates = _near_candidates(value, bands, buckets)
        for previous in candidates:
            if hamming_distance(value, hashes[previous]) <= threshold:
                groups.union(index, previous)
                near_duplicates.union(index, previous)
        representatives[value] = index
        for band, shift, mask in bands:
            buckets[(band, (value >> shift) & mask)].append(index)
    return groups, near_duplicates


def _deduplicate_near_samples(
    samples: list[Sample], hashes: list[int], threshold: int
) -> list[int]:
    """Keep distinct views unless they are directly near an earlier kept image."""
    bands = _hash_band_specs(threshold)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    kept: list[int] = []
    order = sorted(
        range(len(samples)),
        key=lambda index: (samples[index].source != "own", samples[index].path.as_posix()),
    )
    for index in order:
        value = hashes[index]
        if any(
            hamming_distance(value, hashes[previous]) <= threshold
            for previous in _near_candidates(value, bands, buckets)
        ):
            continue
        kept.append(index)
        for band, shift, mask in bands:
            buckets[(band, (value >> shift) & mask)].append(index)
    return sorted(kept)


def _component_key(indices: list[int], samples: list[Sample]) -> str:
    member_keys = sorted(sample_key(samples[index]) for index in indices)
    return hashlib.sha256("\n".join(member_keys).encode()).hexdigest()


def _assign_component_splits(
    components: dict[int, list[int]],
    samples: list[Sample],
    seed: int,
    validation_fraction: float,
    test_fraction: float,
    classes: list[str],
) -> tuple[dict[int, str], dict[str, int]]:
    """Assign one split per duplicate/item component and seed own val/test coverage."""
    forced: dict[int, str] = {}
    own_group_counts: dict[str, int] = Counter(
        sample.target for sample in samples if sample.source == "own"
    )
    own_components: dict[str, list[int]] = {label: [] for label in classes}
    for component, indices in components.items():
        for label in classes:
            if any(samples[index].source == "own" and samples[index].target == label for index in indices):
                own_components[label].append(component)

    def stable_order(component: int) -> str:
        return hashlib.sha256(
            f"{seed}:{_component_key(components[component], samples)}".encode()
        ).hexdigest()

    for label in classes:
        candidates = sorted(own_components[label], key=stable_order)
        available = [component for component in candidates if component not in forced]
        if len(available) >= 2:
            forced[available[0]] = "test"
            forced[available[1]] = "val"
        elif len(available) == 1:
            forced[available[0]] = "test"
            print(
                f"WARNING: own-camera {label} has only one independent group; "
                "it cannot appear in both val and test without leakage."
            )
        elif candidates:
            print(
                f"WARNING: own-camera {label} groups are already tied to another class "
                "or split by near-duplicates; val/test coverage cannot be guaranteed."
            )

    assigned: dict[int, str] = {}
    for component, indices in components.items():
        if component in forced:
            assigned[component] = forced[component]
            continue
        key = _component_key(indices, samples)
        assigned[component] = assign_split(
            key, seed, validation_fraction, test_fraction
        )
    return assigned, own_group_counts


def assign_split(key: str, seed: int, validation_fraction: float, test_fraction: float) -> str:
    """Assign a stable train, val or test split from a grouping key and the seed."""
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    if bucket < test_fraction:
        return "test"
    if bucket < test_fraction + validation_fraction:
        return "val"
    return "train"


def output_stem(sample: Sample, digest: str) -> str:
    """Build a unique output filename stem that keeps an own itemid readable."""
    if sample.item_id is not None:
        return f"own_{sample.item_id}_{digest[:8]}"
    return f"{sample.dataset}_{digest[:12]}"


def write_jpeg(source: Path, destination: Path, max_side: int, quality: int) -> None:
    """Resize a decoded image so its longest side is at most max_side, then save it as JPEG."""
    with Image.open(source) as image:
        image.load()
        # Palette+transparency needs RGBA first or Pillow warns and drops alpha oddly.
        if image.mode in ("P", "PA"):
            image = image.convert("RGBA").convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")
        width, height = image.size
        longest = max(width, height)
        if longest > max_side:
            scale = max_side / longest
            image = image.resize(
                (max(1, round(width * scale)), max(1, round(height * scale))),
                Image.Resampling.LANCZOS,
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, format="JPEG", quality=quality, optimize=True)


def download_hf(cfg: dict[str, Any]) -> Path:
    """Fetch the TrashBusters/combined images into data/raw/ as JPEG files.

    Images are written under their original six labels (cardboard, glass, metal,
    paper, plastic, trash); data/class_map.yaml then maps those onto the model
    classes during the normal run.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "--download-hf needs the `datasets` package; install it with `pip install datasets`."
        ) from exc
    target_root = ROOT / str(cfg["paths"]["raw_root"]) / "combined"
    print(f"downloading {HF_DATASET} into {target_root} ...")
    dataset = load_dataset(HF_DATASET)
    written = 0
    for split_name, shard in dataset.items():
        for index, record in enumerate(shard):
            label = str(record.get("label_name") or record.get("label"))
            destination = target_root / label / f"{split_name}_{index:06d}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            record["image"].convert("RGB").save(destination, format="JPEG", quality=95)
            written += 1
    print(f"wrote {written} images from {HF_DATASET}")
    return target_root


def print_counts(
    counts: dict[str, Counter[tuple[str, str]]],
    classes: list[str],
    minimum: int,
    own_counts: Counter[str],
) -> None:
    """Print class/split/source totals and warn about own-camera collection gaps."""
    sources = ("public", "own")
    header = f"{'class':<10} {'split':<8}" + "".join(f"{source:>10}" for source in sources) + f"{'total':>8}"
    print("\nFinal class x split x source counts:")
    print(header)
    print("-" * len(header))
    for label in classes:
        for split in SPLITS:
            values = [counts[split][(label, source)] for source in sources]
            print(f"{label:<10} {split:<8}" + "".join(f"{value:>10}" for value in values) + f"{sum(values):>8}")
    for label in classes:
        count = own_counts[label]
        if count < minimum:
            print(
                f"!!! WARNING: {label} has only {count} own-camera images; "
                f"collect {minimum - count} more to reach {minimum}."
            )
        for split in ("val", "test"):
            if counts[split][(label, "own")] == 0:
                print(f"!!! WARNING: {split} has no own-camera {label} image.")


def backup_processed(out_root: Path) -> Path | None:
    """Move an existing processed tree into a dated backup rather than deleting it."""
    if not out_root.exists():
        return None
    backup_root = out_root.parent / f"_backup_{datetime.now():%Y-%m-%d}"
    if backup_root.exists() and (backup_root / out_root.name).exists():
        backup_root = out_root.parent / f"_backup_{datetime.now():%Y-%m-%d_%H%M%S}"
    backup_root.mkdir(parents=True, exist_ok=True)
    destination = backup_root / out_root.name
    shutil.move(str(out_root), str(destination))
    print(f"Backed up existing {out_root} to {destination}")
    return destination


def check_leakage(out_root: Path, threshold: int = PHASH_THRESHOLD) -> list[tuple[Path, Path, int]]:
    """Report every train/test pair within the configured perceptual-hash radius."""
    tree = _HashTree()
    train_images = sorted(walk_images(out_root / "train"))
    test_images = sorted(walk_images(out_root / "test"))
    train_hashes: dict[Path, int] = {}
    for path in train_images:
        value = perceptual_hash(path)
        train_hashes[path] = value
        tree.add(value, path)

    matches: list[tuple[Path, Path, int]] = []
    for test_path in test_images:
        test_hash = perceptual_hash(test_path)
        for train_path in tree.within(test_hash, threshold):
            matches.append(
                (test_path, train_path, hamming_distance(test_hash, train_hashes[train_path]))
            )
    print(
        f"\nPerceptual-hash leakage check (dHash threshold <={threshold}; "
        f"{len(test_images)} test x {len(train_images)} train images):"
    )
    if matches:
        for test_path, train_path, distance in matches:
            print(
                f"LEAK: distance={distance} test={test_path.relative_to(ROOT)} "
                f"train={train_path.relative_to(ROOT)}"
            )
    else:
        print("PASS: no test/train image pairs are within the threshold.")
    return matches


def prepare_data(
    clear: bool = False,
    download_hf_dataset: bool = False,
    rebuild: bool = False,
) -> dict[str, Counter[tuple[str, str]]]:
    """Rebuild grouped train/val/test data and a source-aware manifest."""
    cfg = load_config()
    classes = get_classes()
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    paths = cfg["paths"]
    thresholds = cfg["thresholds"]
    max_side = int(thresholds["max_image_side"])
    quality = int(thresholds["jpeg_quality"])
    validation_fraction = float(thresholds["validation_fraction"])
    test_fraction = float(thresholds["test_fraction"])
    class_map = load_class_map(ROOT / str(paths["class_map"]))
    out_root = ROOT / str(paths["processed_root"])

    if rebuild or clear:
        backup_processed(out_root)
    unpack_trashnet(cfg)
    if download_hf_dataset:
        download_hf(cfg)
    else:
        materialize_combined_parquet(cfg)

    print_inventory(cfg, class_map, classes)
    samples, skipped = collect_samples(cfg, class_map, classes)
    for folder in skipped:
        print(f"WARNING: excluding {folder}")
    if skipped:
        print(
            f"WARNING: excluded {len(skipped)} image(s) with empty annotations "
            "or unresolved/invalid labels"
        )

    counts: dict[str, Counter[tuple[str, str]]] = {split: Counter() for split in SPLITS}
    unique: dict[str, tuple[Sample, int]] = {}
    sample_digests: dict[Path, str] = {}
    corrupt = 0
    exact_duplicates = 0
    for sample in samples:
        try:
            payload = sample.path.read_bytes()
        except OSError as exc:
            print(f"WARNING: cannot read {sample.path}: {exc}")
            corrupt += 1
            continue
        digest = hashlib.sha256(payload).hexdigest()
        previous = unique.get(digest)
        if previous is not None:
            exact_duplicates += 1
            if sample.source == "own" and previous[0].source != "own":
                unique[digest] = (sample, previous[1])
                sample_digests[sample.path] = digest
            continue
        try:
            value = perceptual_hash(payload, max_side, quality)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            print(f"WARNING: dropping unreadable image {sample.path}: {exc}")
            corrupt += 1
            continue
        unique[digest] = (sample, value)
        sample_digests[sample.path] = digest

    filtered_samples = [sample for sample, _ in unique.values()]
    hashes = [value for _, value in unique.values()]
    components_union, near_union = _hash_components(
        filtered_samples, hashes, PHASH_THRESHOLD
    )
    components: dict[int, list[int]] = {}
    for index in range(len(filtered_samples)):
        components.setdefault(components_union.find(index), []).append(index)

    assignments, own_counts = _assign_component_splits(
        components, filtered_samples, seed, validation_fraction, test_fraction, classes
    )
    near_components: dict[int, list[int]] = {}
    for index in range(len(filtered_samples)):
        near_components.setdefault(near_union.find(index), []).append(index)

    kept_indices = _deduplicate_near_samples(filtered_samples, hashes, PHASH_THRESHOLD)
    representatives = [
        (index, near_components[near_union.find(index)]) for index in kept_indices
    ]
    near_duplicates = len(filtered_samples) - len(kept_indices)

    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.csv"
    manifest_rows: list[dict[str, str]] = []
    for index, indices in representatives:
        sample = filtered_samples[index]
        split = assignments[components_union.find(index)]
        digest = sample_digests[sample.path]
        destination = out_root / split / sample.target / f"{output_stem(sample, digest)}.jpg"
        try:
            write_jpeg(sample.path, destination, max_side, quality)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            print(f"WARNING: dropping unreadable image {sample.path}: {exc}")
            corrupt += 1
            continue
        own_keys = sorted(
            {sample_key(filtered_samples[i]) for i in indices if filtered_samples[i].source == "own"}
        )
        group = own_keys[0] if len(own_keys) == 1 else (
            f"near:{_component_key(indices, filtered_samples)[:16]}" if len(indices) > 1 else ""
        )
        counts[split][(sample.target, sample.source)] += 1
        manifest_rows.append(
            {
                "path": destination.relative_to(ROOT).as_posix(),
                "class": sample.target,
                "split": split,
                "source": sample.source,
                "group": group,
            }
        )

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "class", "split", "source", "group"))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(
        f"\nread {len(samples)} source image(s): kept {sum(sum(c.values()) for c in counts.values())}, "
        f"dropped {corrupt} corrupt, {exact_duplicates} exact duplicate(s), "
        f"{near_duplicates} perceptual near-duplicate(s)"
    )
    print_counts(counts, classes, MIN_OWN_IMAGES, own_counts)
    class_totals = {
        label: sum(counts[split][(label, source)] for split in SPLITS for source in ("public", "own"))
        for label in classes
    }
    nonzero_totals = [value for value in class_totals.values() if value > 0]
    ratio = max(nonzero_totals) / min(nonzero_totals) if nonzero_totals else float("inf")
    print(f"Class imbalance ratio (largest/smallest non-empty class): {ratio:.2f}:1")
    print(f"Manifest written to {manifest_path.relative_to(ROOT)}")
    check_leakage(out_root)
    return counts


def main() -> None:
    """Parse command-line arguments and build the processed dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-hf",
        action="store_true",
        help=f"download {HF_DATASET} into data/raw/ before preparing (skips local parquet)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="move the existing processed folder to a dated backup before rebuilding",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="back up data/processed to a dated folder, then rebuild from raw and own photos",
    )
    args = parser.parse_args()
    try:
        prepare_data(clear=args.clear, download_hf_dataset=args.download_hf, rebuild=args.rebuild)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
