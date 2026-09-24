"""Image-classification sources, targets, and meta-dataset subsets (paper Appendix C).

Targets and their labeled sources share one class vocabulary:

    USPS, SVHN                                  <- MNIST     (10 digits)
    CIFAR-10.1, CIFAR-10-C                      <- CIFAR-10  (10 classes)
    ImageNet-V2, ImageNet-R, ImageNet-Sketch    <- ImageNet  (ImageNet-1K indices)

ImageNet-family labels always use the official ImageNet-1K index order, i.e. the
sorted WordNet ids of ``<root>/imagenet/val``; folder names are never re-sorted per
target.  ImageNet-R covers 200 of the 1000 classes, so predictions are restricted to
the classes present in the target.

Meta-dataset subsets follow the AutoEval synthetic-shift recipe: each subset samples
images from the labeled source test split and applies one synthetic shift family at
a random severity.  Every subset is a deterministic function of (seed, subset index).
"""

from __future__ import annotations

import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


DIGIT_NAMES = [str(i) for i in range(10)]
CIFAR10_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck",
]
TEMPLATES = {
    "digits": 'a photo of the number: "{}".',
    "cifar10": "a photo of a {}.",
    "imagenet": "a photo of a {}.",
}
DOMAINS = {
    "digits": "handwritten or photographed digit",
    "cifar10": "natural image",
    "imagenet": "natural image",
}
# target -> (source, vocabulary)
IMAGE_TARGETS = {
    "usps": ("mnist", "digits"),
    "svhn": ("mnist", "digits"),
    "cifar10.1": ("cifar10", "cifar10"),
    "cifar10-c": ("cifar10", "cifar10"),
    "imagenet-v2": ("imagenet", "imagenet"),
    "imagenet-r": ("imagenet", "imagenet"),
    "imagenet-sketch": ("imagenet", "imagenet"),
}
CIFAR10C_CORRUPTIONS = [
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
    "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression",
    "motion_blur", "pixelate", "saturate", "shot_noise", "snow", "spatter",
    "speckle_noise", "zoom_blur",
]
CIFAR101_URL = "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_{}.npy"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageSet:
    """Lazily loaded labeled images; ``images[i]`` returns an RGB PIL image."""

    def __init__(self, name: str, loader: Callable[[int], Any], labels: Sequence[int]):
        self.name = name
        self._loader = loader
        self.labels = np.asarray(labels, dtype=int)

    def __len__(self) -> int:
        return int(self.labels.size)

    def image(self, index: int) -> Any:
        return self._loader(int(index)).convert("RGB")

    def images(self, indices: Sequence[int]) -> list[Any]:
        return [self.image(index) for index in indices]


def _torchvision(name: str, dataset: Any, labels: Sequence[int]) -> ImageSet:
    return ImageSet(name, lambda index: dataset[index][0], labels)


def _array(name: str, images: np.ndarray, labels: Sequence[int]) -> ImageSet:
    from PIL import Image

    return ImageSet(name, lambda index: Image.fromarray(images[index]), labels)


def _files(name: str, paths: Sequence[Path], labels: Sequence[int]) -> ImageSet:
    from PIL import Image

    return ImageSet(name, lambda index: Image.open(paths[index]), labels)


def _download_unverified(fn: Callable[[], Any]) -> Any:
    # The USPS mirror used by torchvision serves an incomplete certificate chain.
    context = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        return fn()
    finally:
        ssl._create_default_https_context = context


def _class_folders(root: Path) -> list[Path]:
    """Class folders under ``root``, descending through single wrapper directories."""
    if not root.is_dir():
        raise FileNotFoundError(f"missing image directory: {root}")
    while True:
        folders = sorted(child for child in root.iterdir() if child.is_dir())
        if len(folders) != 1:
            return folders
        root = folders[0]


def imagenet_wnids(root: str | Path) -> list[str]:
    """The 1000 WordNet ids in official ImageNet-1K index order."""
    base = Path(root) / "imagenet"
    for split in ("val", "train"):
        if (base / split).is_dir():
            wnids = sorted(folder.name for folder in _class_folders(base / split))
            if len(wnids) != 1000:
                raise ValueError(f"{base / split} has {len(wnids)} class folders, expected 1000")
            return wnids
    raise FileNotFoundError(f"ImageNet class folders not found under {base}/val or {base}/train")


def imagenet_names() -> list[str]:
    from torchvision.models import ResNet50_Weights

    return list(ResNet50_Weights.IMAGENET1K_V2.meta["categories"])


def class_names(vocabulary: str) -> list[str]:
    if vocabulary == "digits":
        return DIGIT_NAMES
    if vocabulary == "cifar10":
        return CIFAR10_NAMES
    if vocabulary == "imagenet":
        return imagenet_names()
    raise KeyError(vocabulary)


def _folder_set(name: str, folders: Sequence[Path], label_of: Callable[[str], int]) -> ImageSet:
    paths: list[Path] = []
    labels: list[int] = []
    for folder in folders:
        label = label_of(folder.name)
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in _IMAGE_SUFFIXES:
                paths.append(path)
                labels.append(label)
    if not paths:
        raise FileNotFoundError(f"no images found for {name}")
    return _files(name, paths, labels)


def load_images(name: str, split: str, root: str | Path, settings: Mapping[str, Any] | None = None) -> ImageSet:
    """Load a source split ("train"/"test") or a target dataset (split ignored)."""
    import torchvision

    root = Path(root)
    settings = settings or {}
    train = split == "train"
    if name == "mnist":
        dataset = torchvision.datasets.MNIST(root / "mnist", train=train, download=True)
        return _torchvision(f"mnist-{split}", dataset, dataset.targets.tolist())
    if name == "usps":
        dataset = _download_unverified(
            lambda: torchvision.datasets.USPS(root / "usps", train=False, download=True)
        )
        return _torchvision("usps-test", dataset, list(dataset.targets))
    if name == "svhn":
        dataset = torchvision.datasets.SVHN(root / "svhn", split="test", download=True)
        return _torchvision("svhn-test", dataset, dataset.labels.tolist())
    if name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root / "cifar10", train=train, download=True)
        return _torchvision(f"cifar10-{split}", dataset, list(dataset.targets))
    if name == "cifar10.1":
        directory = root / "cifar10.1"
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for part in ("data", "labels"):
            path = directory / f"cifar10.1_v6_{part}.npy"
            if not path.exists():
                from urllib.request import urlretrieve

                urlretrieve(CIFAR101_URL.format(part), path)
            arrays[part] = np.load(path)
        return _array("cifar10.1", arrays["data"], arrays["labels"].astype(int))
    if name == "cifar10-c":
        directory = root / "cifar10-c"
        if (directory / "CIFAR-10-C").is_dir():
            directory = directory / "CIFAR-10-C"
        corruptions = settings.get("corruptions", "all")
        corruptions = CIFAR10C_CORRUPTIONS if corruptions == "all" else list(corruptions)
        severities = [int(s) for s in settings.get("severities", [1, 2, 3, 4, 5])]
        labels_all = np.load(directory / "labels.npy").astype(int)
        blocks: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for corruption in corruptions:
            data = np.load(directory / f"{corruption}.npy", mmap_mode="r")
            for severity in severities:
                rows = slice((severity - 1) * 10000, severity * 10000)
                blocks.append(np.asarray(data[rows]))
                labels.append(labels_all[rows])
        return _array("cifar10-c", np.concatenate(blocks), np.concatenate(labels))
    if name == "imagenet":
        wnids = imagenet_wnids(root)
        index = {wnid: i for i, wnid in enumerate(wnids)}
        folder = root / "imagenet" / ("train" if train else "val")
        return _folder_set(f"imagenet-{split}", _class_folders(folder), index.__getitem__)
    if name == "imagenet-v2":
        # ImageNetV2 names its class folders by ImageNet-1K index (0..999).
        return _folder_set("imagenet-v2", _class_folders(root / "imagenet-v2"), int)
    if name in {"imagenet-r", "imagenet-sketch"}:
        index = {wnid: i for i, wnid in enumerate(imagenet_wnids(root))}
        return _folder_set(name, _class_folders(root / name), index.__getitem__)
    raise KeyError(f"unknown image dataset {name!r}")


@dataclass(frozen=True)
class MetaSubset:
    """One labeled meta-dataset subset P_r: source-test indices plus a synthetic shift."""

    index: int
    indices: tuple[int, ...]
    shift: str
    severity: float
    seed: int


SHIFTS = (
    "rotate", "color", "blur", "noise", "perspective",
    "contrast", "posterize", "invert", "sharpness", "occlusion",
)


def make_meta_subsets(source_test: ImageSet, count: int, size: int, seed: int) -> list[MetaSubset]:
    subsets: list[MetaSubset] = []
    for r in range(int(count)):
        rng = np.random.default_rng([seed, r])
        chosen = rng.choice(len(source_test), size=min(int(size), len(source_test)), replace=False)
        subsets.append(
            MetaSubset(
                index=r,
                indices=tuple(int(i) for i in chosen),
                shift=SHIFTS[r % len(SHIFTS)],
                severity=float(rng.uniform(0.1, 1.0)),
                seed=int(seed),
            )
        )
    return subsets


def apply_shift(image: Any, shift: str, severity: float, rng: np.random.Generator) -> Any:
    """Apply one synthetic shift family to an RGB PIL image."""
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps

    s = float(severity)
    if shift == "rotate":
        return image.rotate(float(rng.uniform(-1.0, 1.0)) * 45.0 * s, resample=Image.BILINEAR)
    if shift == "color":
        for enhance in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
            image = enhance(image).enhance(max(0.05, 1.0 + float(rng.uniform(-0.8, 0.8)) * s))
        return image
    if shift == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=3.0 * s * float(rng.uniform(0.5, 1.0))))
    if shift == "noise":
        array = np.asarray(image, dtype=np.float32)
        array = array + rng.normal(0.0, 60.0 * s, size=array.shape)
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    if shift == "perspective":
        width, height = image.size
        d = 0.5 * s
        jitter = lambda limit: float(rng.uniform(0.0, d * limit))  # noqa: E731
        quad = (
            jitter(width), jitter(height),
            jitter(width), height - jitter(height),
            width - jitter(width), height - jitter(height),
            width - jitter(width), jitter(height),
        )
        return image.transform(image.size, Image.QUAD, quad, resample=Image.BILINEAR)
    if shift == "contrast":
        return ImageEnhance.Contrast(image).enhance(1.0 - 0.9 * s * float(rng.uniform(0.5, 1.0)))
    if shift == "posterize":
        return ImageOps.posterize(image, max(1, 8 - int(round(6 * s))))
    if shift == "invert":
        return ImageOps.invert(image) if rng.uniform() < s else image
    if shift == "sharpness":
        return ImageEnhance.Sharpness(image).enhance(1.0 + 8.0 * s * float(rng.uniform(0.5, 1.0)))
    if shift == "occlusion":
        width, height = image.size
        side = max(1, int(np.sqrt(0.5 * s) * min(width, height)))
        x0 = int(rng.integers(0, max(1, width - side)))
        y0 = int(rng.integers(0, max(1, height - side)))
        image = image.copy()
        image.paste((127, 127, 127), (x0, y0, x0 + side, y0 + side))
        return image
    raise KeyError(f"unknown shift {shift!r}")


def subset_images(source_test: ImageSet, subset: MetaSubset) -> list[Any]:
    images: list[Any] = []
    for k, index in enumerate(subset.indices):
        rng = np.random.default_rng([subset.seed, subset.index, k])
        images.append(apply_shift(source_test.image(index), subset.shift, subset.severity, rng))
    return images


def subset_labels(source_test: ImageSet, subset: MetaSubset) -> np.ndarray:
    return source_test.labels[list(subset.indices)]
