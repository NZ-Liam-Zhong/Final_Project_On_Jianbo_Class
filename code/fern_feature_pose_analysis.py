#!/usr/bin/env python
"""Analyze OpenVLA language features against pose similarities on the Fern dataset."""

from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


HF_CACHE_DIR = Path("/mnt/disk1/ilykyleliam/public")
PROMPT = "In: What action should the robot take to push the trees away from the wall?\nOut:"


def ensure_hf_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir))


def iter_images(image_dir: Path) -> Iterable[Path]:
    supported = {".jpg", ".jpeg", ".png"}
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() in supported:
            yield path


def load_pose_translations(poses_bounds_path: Path) -> torch.Tensor:
    poses_bounds = np.load(poses_bounds_path)
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)
    translations = poses[:, :, 3]
    return torch.from_numpy(translations).to(torch.float32)


def rbf_affinity_from_features(
    features: torch.Tensor,
    features_B: torch.Tensor | None = None,
    gamma: float = 1.0,
) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError("`features` must be a 2D tensor [n_samples, dim].")

    features_B = features if features_B is None else features_B
    d = torch.cdist(features, features_B, p=2)
    squared = torch.pow(d, 2)
    sigma = 2 * gamma * features.var(dim=0, unbiased=False).sum()
    sigma = torch.clamp(sigma, min=1e-6)
    return torch.exp(-squared / sigma)


def affinity_signature_distances(affinity: torch.Tensor) -> torch.Tensor:
    if affinity.ndim != 2:
        raise ValueError("`affinity` must be a 2D tensor.")

    signatures = torch.nn.functional.normalize(affinity.to(torch.float32), dim=1)
    similarities = signatures @ signatures.t()
    similarities = similarities.clamp(-1.0, 1.0)
    return 1.0 - similarities


def classical_mds(distances: torch.Tensor, output_dim: int = 2) -> np.ndarray:
    if distances.ndim != 2:
        raise ValueError("`distances` must be a 2D tensor.")

    dist_np = distances.cpu().numpy()
    sq_dist = dist_np**2
    n = sq_dist.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ sq_dist @ centering
    eigvals, eigvecs = np.linalg.eigh(gram)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    positive = eigvals > 0
    eigvals = eigvals[positive][:output_dim]
    eigvecs = eigvecs[:, positive][:, : output_dim]
    coords = eigvecs * np.sqrt(np.maximum(eigvals, 0))
    if coords.shape[1] < output_dim:
        padding = np.zeros((coords.shape[0], output_dim - coords.shape[1]))
        coords = np.hstack([coords, padding])
    return coords


def collect_language_features(
    model: AutoModelForVision2Seq,
    processor: AutoProcessor,
    image_paths: Iterable[Path],
    prompt: str,
    device: torch.device,
) -> torch.Tensor:
    feature_list = []
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    for idx, image_path in enumerate(image_paths):
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        inputs = processor(
            text=[prompt],
            images=image,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}

        with torch.no_grad():
            with autocast_ctx:
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    output_attentions=False,
                    output_projector_features=True,
                    return_dict=True,
                )

        last_hidden = outputs.hidden_states[-1][0].to(torch.float32)
        feature_list.append(last_hidden.mean(dim=0).cpu())

    return torch.stack(feature_list, dim=0)


def plot_embeddings(
    feature_coords: np.ndarray,
    pose_coords: np.ndarray,
    output_path: Path,
) -> None:
    num_samples = feature_coords.shape[0]
    indices = np.arange(num_samples)
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    scatter_kwargs = dict(s=70, edgecolor="black", linewidth=0.5, cmap=cmap, c=indices)

    axes[0].scatter(feature_coords[:, 0], feature_coords[:, 1], **scatter_kwargs)
    axes[0].set_title("Language Feature Affinity (K-Means Distances)")
    axes[0].set_xlabel("Center 0 distance")
    axes[0].set_ylabel("Center 1 distance")
    axes[0].set_aspect("equal")

    axes[1].scatter(pose_coords[:, 0], pose_coords[:, 1], **scatter_kwargs)
    axes[1].set_title("Pose Affinity (K-Means Distances)")
    axes[1].set_xlabel("Center 0 distance")
    axes[1].set_ylabel("Center 1 distance")
    axes[1].set_aspect("equal")

    for ax, coords in zip(axes, (feature_coords, pose_coords)):
        for idx, (x_val, y_val) in enumerate(coords):
            ax.text(
                x_val,
                y_val,
                str(idx),
                fontsize=9,
                ha="center",
                va="center",
                color="white",
                weight="bold",
            )

    fig.suptitle("Fern Dataset: Feature vs Pose Similarities", fontsize=14)
    fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap),
        ax=axes,
        fraction=0.02,
        pad=0.04,
        label="Image index",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_hf_cache(HF_CACHE_DIR)

    repo_root = Path(__file__).resolve().parent.parent
    fern_dir = repo_root / "fern"
    image_dir = fern_dir / "images"
    poses_bounds_path = fern_dir / "poses_bounds.npy"
    output_path = fern_dir / "fern_feature_pose_affinity.png"

    image_paths = list(iter_images(image_dir))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    if not poses_bounds_path.exists():
        raise FileNotFoundError(f"Missing pose file: {poses_bounds_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()

    language_features = collect_language_features(model, processor, image_paths, PROMPT, device)
    pose_translations = load_pose_translations(poses_bounds_path)

    feature_affinity = rbf_affinity_from_features(language_features)
    pose_affinity = rbf_affinity_from_features(pose_translations)

    feature_coords, _ = kmeans_projection(feature_affinity)
    pose_coords, _ = kmeans_projection(pose_affinity)

    plot_embeddings(feature_coords, pose_coords, output_path)
    print(f"Saved comparison figure to: {output_path}")


if __name__ == "__main__":
    main()

