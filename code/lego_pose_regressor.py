#!/usr/bin/env python
"""Train a lightweight regressor that maps OpenVLA encoder features to Lego camera poses."""

from __future__ import annotations

import argparse
import gc
import json
import random
from dataclasses import dataclass
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor

from scripts.fern_feature_pose_analysis import (
    HF_CACHE_DIR,
    PROMPT,
    _resolve_lego_frame_path,
    ensure_hf_cache,
    get_device_and_precision,
)


@dataclass
class LegoSplit:
    label: str
    image_paths: List[Path]
    pose_matrices: torch.Tensor  # (N, 3, 4)


class FeaturePoseDataset(Dataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        if len(features) != len(targets):
            raise ValueError("Features and targets must have same length")
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


class PoseRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.LayerNorm(input_dim)]
        in_dim = input_dim
        for _ in range(max(num_hidden_layers, 0)):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, target_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lego pose regression from OpenVLA features")
    parser.add_argument("--train-split", default="train", help="Lego split used for training")
    parser.add_argument("--test-split", default="test", help="Lego split used for evaluation")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap for train samples")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional cap for test samples")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=32, help="Eval batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clip (<=0 to disable)")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension for MLP")
    parser.add_argument("--num-hidden-layers", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--prompt", default=PROMPT, help="Language prompt passed to the processor")
    parser.add_argument("--model-name", default="openvla/openvla-7b", help="HF model identifier")
    parser.add_argument("--feature-pooling", choices=["mean", "mean_std"], default="mean_std", help="Pooling strategy")
    parser.add_argument("--output-dir", type=Path, default=Path("lego/pose_regression_outputs"), help="Directory to store metrics")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_lego_split(repo_root: Path, split: str, max_samples: Optional[int]) -> LegoSplit:
    lego_dir = repo_root / "lego"
    meta_path = lego_dir / f"transforms_{split}.json"
    split_dir = lego_dir / split
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    frames = meta["frames"]
    if max_samples is not None:
        frames = frames[:max_samples]

    image_paths: List[Path] = []
    pose_mats: List[torch.Tensor] = []

    for frame in frames:
        resolved = _resolve_lego_frame_path(lego_dir, frame["file_path"])
        if not resolved.exists():
            resolved = split_dir / Path(frame["file_path"]).name
        if not resolved.exists():
            raise FileNotFoundError(f"Could not resolve image path for split '{split}': {frame['file_path']}")
        image_paths.append(resolved)
        matrix = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        pose_mats.append(matrix[:3, :4])

    pose_tensor = torch.stack(pose_mats)
    return LegoSplit(label=f"Lego ({split})", image_paths=image_paths, pose_matrices=pose_tensor)


def pool_hidden_states(hidden: torch.Tensor, strategy: str) -> torch.Tensor:
    if hidden.ndim != 2:
        raise ValueError("Expected hidden state tensor of shape (seq_len, hidden_dim)")
    if strategy == "mean":
        return hidden.mean(dim=0)
    if strategy == "mean_std":
        mean = hidden.mean(dim=0)
        std = hidden.std(dim=0, unbiased=False)
        return torch.cat([mean, std], dim=0)
    raise ValueError(f"Unsupported pooling strategy: {strategy}")


def collect_pooled_features(
    model: AutoModelForVision2Seq,
    processor: AutoProcessor,
    image_paths: Sequence[Path],
    prompt: str,
    device: torch.device,
    autocast_dtype: Optional[torch.dtype],
    pooling: str,
) -> torch.Tensor:
    pooled_features: List[torch.Tensor] = []
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if (device.type == "cuda" and autocast_dtype is not None)
        else nullcontext()
    )
    with torch.no_grad():
        for image_path in image_paths:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
            inputs = processor(text=[prompt], images=image, return_tensors="pt")
            inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            with autocast_ctx:
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    output_attentions=False,
                    output_projector_features=True,
                    return_dict=True,
                )
            last_hidden = outputs.hidden_states[-1][0].to(torch.float32).cpu()
            pooled_features.append(pool_hidden_states(last_hidden, pooling))
    return torch.stack(pooled_features)


def flatten_pose_matrices(pose_matrices: torch.Tensor) -> torch.Tensor:
    if pose_matrices.ndim != 3 or pose_matrices.shape[1:] != (3, 4):
        raise ValueError("Pose matrices must have shape (N, 3, 4)")
    return pose_matrices.reshape(pose_matrices.shape[0], -1)


def make_dataloader(
    features: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
) -> DataLoader[Tuple[torch.Tensor, torch.Tensor]]:
    dataset = FeaturePoseDataset(features, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_epoch(
    model: PoseRegressor,
    dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    mse_loss = nn.MSELoss()
    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device)
        preds = model(features)
        loss = mse_loss(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * features.size(0)
        total_samples += features.size(0)
    return total_loss / max(total_samples, 1)


def evaluate_mse(
    model: PoseRegressor,
    features: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> Tuple[float, torch.Tensor]:
    model.eval()
    mse_sum = 0.0
    total = 0
    criterion = nn.MSELoss(reduction="sum")
    preds: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            end = start + batch_size
            batch_feats = features[start:end].to(device)
            batch_targets = targets[start:end].to(device)
            batch_preds = model(batch_feats)
            mse_sum += criterion(batch_preds, batch_targets).item()
            total += batch_preds.shape[0]
            preds.append(batch_preds.cpu())
    stacked_preds = torch.cat(preds, dim=0) if preds else torch.empty_like(targets)
    mse = mse_sum / max(total, 1)
    return mse, stacked_preds


def save_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    test_mse: float,
    train_eval_mse: float,
    train_loss_history: List[float],
    train_size: int,
    test_size: int,
    test_targets: torch.Tensor,
    test_predictions: torch.Tensor,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "test_mse": test_mse,
        "train_eval_mse": train_eval_mse,
        "train_loss_history": train_loss_history,
        "train_samples": train_size,
        "test_samples": test_size,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "hidden_dim": args.hidden_dim,
            "num_hidden_layers": args.num_hidden_layers,
            "dropout": args.dropout,
            "prompt": args.prompt,
            "model_name": args.model_name,
            "feature_pooling": args.feature_pooling,
        },
    }
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    npz_path = output_dir / "test_predictions_vs_targets.npz"
    np.savez(
        npz_path,
        test_predictions=test_predictions.numpy(),
        test_targets=test_targets.numpy(),
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_hf_cache(HF_CACHE_DIR)

    repo_root = Path(__file__).resolve().parent.parent
    train_split = load_lego_split(repo_root, args.train_split, args.max_train_samples)
    test_split = load_lego_split(repo_root, args.test_split, args.max_test_samples)

    if len(train_split.image_paths) == 0 or len(test_split.image_paths) == 0:
        raise RuntimeError("Both train and test splits must contain at least one sample.")

    device, model_dtype, autocast_dtype = get_device_and_precision()
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    encoder_model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)
    encoder_model.eval()

    combined_paths = train_split.image_paths + test_split.image_paths
    all_features = collect_pooled_features(
        encoder_model,
        processor,
        combined_paths,
        args.prompt,
        device,
        autocast_dtype,
        pooling=args.feature_pooling,
    )
    del encoder_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    train_count = len(train_split.image_paths)
    train_features = all_features[:train_count]
    test_features = all_features[train_count:]

    train_targets = flatten_pose_matrices(train_split.pose_matrices)
    test_targets = flatten_pose_matrices(test_split.pose_matrices)

    model = PoseRegressor(
        input_dim=train_features.shape[1],
        target_dim=train_targets.shape[1],
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataloader = make_dataloader(train_features, train_targets, args.batch_size)

    train_loss_history: List[float] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, dataloader, optimizer, device, args.grad_clip)
        train_loss_history.append(train_loss)
        print(f"Epoch {epoch:03d} | train_mse={train_loss:.6f}")

    train_eval_mse, _ = evaluate_mse(
        model,
        train_features,
        train_targets,
        device,
        args.eval_batch_size,
    )
    print(f"Train MSE ({args.train_split}): {train_eval_mse:.6f}")

    test_mse, test_predictions = evaluate_mse(
        model,
        test_features,
        test_targets,
        device,
        args.eval_batch_size,
    )
    print(f"Test MSE ({args.test_split}): {test_mse:.6f}")

    save_outputs(
        args.output_dir,
        args,
        test_mse,
        train_eval_mse,
        train_loss_history,
        train_features.shape[0],
        test_features.shape[0],
        test_targets,
        test_predictions,
    )


if __name__ == "__main__":
    main()


