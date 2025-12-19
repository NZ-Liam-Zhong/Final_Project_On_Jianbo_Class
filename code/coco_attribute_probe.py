#!/usr/bin/env python
"""Linear attribute probing on COCO captions using OpenVLA features."""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor

from scripts.lego_pose_regressor import (
    HF_CACHE_DIR,
    collect_pooled_features,
    ensure_hf_cache,
    get_device_and_precision,
    set_seed,
)


COCO_IMAGES_SUBDIR = "val2017"
COCO_CAPTIONS_FILE = "annotations/captions_val2017.json"

DEFAULT_ATTRIBUTES: Dict[str, List[str]] = {
    "red": ["red"],
    "blue": ["blue"],
    "green": ["green"],
    "yellow": ["yellow"],
    "orange": ["orange"],
    "purple": ["purple"],
    "pink": ["pink"],
    "black": ["black", "dark"],
    "white": ["white"],
    "brown": ["brown"],
    "gray": ["gray", "grey", "silver"],
    "striped": ["striped", "stripes"],
    "spotted": ["spotted", "polka"],
    "wooden": ["wood", "wooden"],
    "metal": ["metal", "metallic", "steel"],
    "plastic": ["plastic"],
    "round": ["round", "circular"],
    "square": ["square", "boxy"],
    "shiny": ["shiny", "glossy"],
}

TOKEN_PATTERN = re.compile(r"[a-z]+")


@dataclass
class ImageEntry:
    image_id: int
    path: Path
    labels: torch.Tensor


class MultiLabelDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COCO attribute probing with OpenVLA features")
    parser.add_argument(
        "--coco-root",
        type=Path,
        default=Path("/mnt/disk1/ilykyleliam/public/datasets/coco2017"),
        help="Path to COCO2017 directory containing val2017/ and annotations/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/coco_attribute_probe"),
        help="Directory for metrics and caches",
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=Path("experiments/coco_attribute_probe/val_openvla_features.pt"),
        help="Optional cache for extracted OpenVLA features",
    )
    parser.add_argument("--max-images", type=int, default=1200, help="Max number of images to sample")
    parser.add_argument("--min-positives", type=int, default=40, help="Minimum positives required to keep attribute")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio (rest test)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for probe training")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for probe training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for probe")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for probe optimizer")
    parser.add_argument("--prompt", type=str, default="In: Describe the scene.\nOut:", help="Prompt used for OpenVLA")
    parser.add_argument("--model-name", type=str, default="openvla/openvla-7b", help="Hugging Face model id")
    parser.add_argument(
        "--feature-pooling",
        choices=["mean", "mean_std"],
        default="mean_std",
        help="Pooling strategy for hidden states",
    )
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    return parser.parse_args()


def load_captions(annotation_path: Path) -> Tuple[List[dict], Dict[int, List[str]]]:
    with annotation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    id_to_captions: Dict[int, List[str]] = {}
    for ann in data["annotations"]:
        id_to_captions.setdefault(ann["image_id"], []).append(ann["caption"])
    return data["images"], id_to_captions


def tokenize_caption(caption: str) -> List[str]:
    return TOKEN_PATTERN.findall(caption.lower())


def build_attribute_vectors(
    image_records: List[dict],
    id_to_captions: Dict[int, List[str]],
    coco_root: Path,
    max_images: int,
    min_positives: int,
) -> Tuple[List[ImageEntry], List[str]]:
    rng = random.Random(0)
    entries: List[ImageEntry] = []
    attr_names = list(DEFAULT_ATTRIBUTES.keys())

    for record in image_records:
        image_id = record["id"]
        captions = id_to_captions.get(image_id)
        if not captions:
            continue
        tokens = set()
        for caption in captions:
            tokens.update(tokenize_caption(caption))
        label_vec = [int(any(keyword in tokens for keyword in DEFAULT_ATTRIBUTES[name])) for name in attr_names]
        if sum(label_vec) == 0:
            continue
        img_path = coco_root / COCO_IMAGES_SUBDIR / record["file_name"]
        if not img_path.exists():
            continue
        entries.append(ImageEntry(image_id=image_id, path=img_path, labels=torch.tensor(label_vec, dtype=torch.float32)))

    rng.shuffle(entries)
    if max_images is not None and max_images < len(entries):
        entries = entries[:max_images]

    stacked = torch.stack([entry.labels for entry in entries], dim=0)
    keep_mask = (stacked.sum(dim=0) >= min_positives).numpy()
    kept_indices = [idx for idx, keep in enumerate(keep_mask) if keep]
    filtered_attr_names = [attr_names[idx] for idx in kept_indices]
    filtered_entries: List[ImageEntry] = []
    for entry in entries:
        filtered_entries.append(
            ImageEntry(
                image_id=entry.image_id,
                path=entry.path,
                labels=entry.labels[kept_indices],
            )
        )
    return filtered_entries, filtered_attr_names


def load_or_create_feature_cache(
    entries: Sequence[ImageEntry],
    args: argparse.Namespace,
) -> torch.Tensor:
    cache_path = args.feature_cache
    if cache_path.exists():
        payload = torch.load(cache_path)
        if (
            payload.get("prompt") == args.prompt
            and payload.get("feature_pooling") == args.feature_pooling
            and payload.get("model_name") == args.model_name
            and payload.get("image_ids") == [entry.image_id for entry in entries]
        ):
            return payload["features"]
        print(f"Feature cache {cache_path} exists but metadata mismatch; recomputing.")

    ensure_hf_cache(HF_CACHE_DIR)
    device, model_dtype, autocast_dtype = get_device_and_precision()
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    encoder = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)
    encoder.eval()

    features = collect_pooled_features(
        encoder,
        processor,
        [entry.path for entry in entries],
        args.prompt,
        device,
        autocast_dtype,
        pooling=args.feature_pooling,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features,
            "image_ids": [entry.image_id for entry in entries],
            "prompt": args.prompt,
            "feature_pooling": args.feature_pooling,
            "model_name": args.model_name,
        },
        cache_path,
    )
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return features


def split_indices(num_samples: int, train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    train_end = int(num_samples * train_ratio)
    val_end = int(num_samples * (train_ratio + val_ratio))
    train_idx = indices[:train_end].tolist()
    val_idx = indices[train_end:val_end].tolist()
    test_idx = indices[val_end:].tolist()
    return train_idx, val_idx, test_idx


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    pos_total = y_true.sum()
    if pos_total == 0:
        return float("nan")
    order = np.argsort(-scores)
    sorted_true = y_true[order]
    cumulative = np.cumsum(sorted_true)
    precision = cumulative / (np.arange(len(sorted_true)) + 1)
    ap = (precision * sorted_true).sum() / pos_total
    return float(ap)


def f1_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> float:
    preds = scores >= threshold
    tp = np.logical_and(preds, y_true).sum()
    fp = np.logical_and(preds, np.logical_not(y_true)).sum()
    fn = np.logical_and(np.logical_not(preds), y_true).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def train_probe(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    num_labels: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[nn.Module, List[float]]:
    model = nn.Linear(input_dim, num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    history: List[float] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        total = 0
        for feats, labels in train_loader:
            feats = feats.to(device)
            labels = labels.to(device)
            logits = model(feats)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * feats.size(0)
            total += feats.size(0)
        epoch_loss /= max(total, 1)
        history.append(epoch_loss)
        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            val_loss = evaluate_loss(model, val_loader, criterion, device)
            print(f"Epoch {epoch:03d} | train_loss={epoch_loss:.4f} | val_loss={val_loss:.4f}")
    return model, history


def evaluate_loss(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for feats, labels in dataloader:
            feats = feats.to(device)
            labels = labels.to(device)
            logits = model(feats)
            loss = criterion(logits, labels)
            total += loss.item() * feats.size(0)
            count += feats.size(0)
    return total / max(count, 1)


def collect_logits(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for feats, labels in dataloader:
            feats = feats.to(device)
            logits = model(feats).cpu()
            logits_list.append(logits)
            labels_list.append(labels)
    return torch.cat(logits_list).numpy(), torch.cat(labels_list).numpy()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    coco_root = args.coco_root
    annotations_path = coco_root / COCO_CAPTIONS_FILE
    if not annotations_path.exists():
        raise FileNotFoundError(f"Missing COCO captions at {annotations_path}")

    images, id_to_captions = load_captions(annotations_path)
    image_entries, attr_names = build_attribute_vectors(
        images,
        id_to_captions,
        coco_root,
        args.max_images,
        args.min_positives,
    )
    if not image_entries:
        raise RuntimeError("No eligible images after filtering.")

    labels_tensor = torch.stack([entry.labels for entry in image_entries])
    features = load_or_create_feature_cache(image_entries, args)

    train_idx, val_idx, test_idx = split_indices(len(image_entries), args.train_ratio, args.val_ratio, args.seed)
    feature_tensor = features.clone().to(torch.float32)

    train_dataset = MultiLabelDataset(feature_tensor[train_idx], labels_tensor[train_idx])
    val_dataset = MultiLabelDataset(feature_tensor[val_idx], labels_tensor[val_idx])
    test_dataset = MultiLabelDataset(feature_tensor[test_idx], labels_tensor[test_idx])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_history = train_probe(
        train_loader,
        val_loader,
        input_dim=feature_tensor.shape[1],
        num_labels=len(attr_names),
        args=args,
        device=device,
    )

    test_logits, test_labels = collect_logits(model, test_loader, device)
    test_probs = 1 / (1 + np.exp(-test_logits))
    per_attr_metrics = []
    aps = []
    f1s = []
    positives = test_labels.sum(axis=0)
    for idx, attr in enumerate(attr_names):
        y_true = test_labels[:, idx]
        scores = test_probs[:, idx]
        ap = average_precision(y_true, scores)
        f1 = f1_at_threshold(y_true, scores)
        per_attr_metrics.append(
            {
                "attribute": attr,
                "positives": int(positives[idx]),
                "average_precision": ap,
                "f1_at_0_5": f1,
            }
        )
        if not np.isnan(ap):
            aps.append(ap)
        f1s.append(f1)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "attribute_probe_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "mean_average_precision": float(np.nanmean(aps)) if aps else None,
                "mean_f1_at_0_5": float(np.mean(f1s)) if f1s else None,
                "train_loss_history": train_history,
                "attributes": per_attr_metrics,
                "config": {
                    "max_images": args.max_images,
                    "train_ratio": args.train_ratio,
                    "val_ratio": args.val_ratio,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "prompt": args.prompt,
                    "model_name": args.model_name,
                    "feature_pooling": args.feature_pooling,
                },
            },
            f,
            indent=2,
        )
    print(f"Saved attribute probe metrics to {metrics_path}")


