#!/usr/bin/env python
"""Prompt sensitivity analysis for OpenVLA features on LEGO/Fern datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from scripts.fern_feature_pose_analysis import PROMPT as BASE_PROMPT
from scripts.fern_feature_pose_analysis import load_lego_val_dataset, load_fern_dataset
from scripts.lego_pose_regressor import (
    HF_CACHE_DIR,
    collect_pooled_features,
    ensure_hf_cache,
    get_device_and_precision,
    set_seed,
)

PROMPT_VARIANTS = {
    "base": BASE_PROMPT,
    "verb_swap": BASE_PROMPT.replace("push the trees away from the wall", "pull the trees towards the robot"),
    "noun_swap": BASE_PROMPT.replace("trees", "boxes"),
    "neutral": "In: Describe the scene in detail.\nOut:",
    "short": "In: Action?\nOut:",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prompt perturbation sensitivity for OpenVLA features")
    parser.add_argument(
        "--dataset",
        choices=["lego_val", "fern"],
        default="lego_val",
        help="Dataset to analyze",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=120,
        help="Maximum number of frames to process",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/prompt_perturbation"),
        help="Directory for analysis outputs",
    )
    parser.add_argument("--model-name", type=str, default="openvla/openvla-7b", help="HF model name")
    parser.add_argument("--feature-pooling", choices=["mean", "mean_std"], default="mean_std")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    return parser.parse_args()


def load_dataset_bundle(repo_root: Path, dataset: str, max_images: int):
    if dataset == "lego_val":
        bundle = load_lego_val_dataset(repo_root, max_images)
    elif dataset == "fern":
        bundle = load_fern_dataset(repo_root, max_images)
    else:
        raise ValueError(f"Unsupported dataset {dataset}")
    if max_images is not None and max_images < len(bundle.image_paths):
        bundle.image_paths = bundle.image_paths[:max_images]
    return bundle


def summarize_differences(base: np.ndarray, variant: np.ndarray) -> Dict[str, float]:
    diff = variant - base
    base_norm = np.linalg.norm(base, axis=1) + 1e-6
    variant_norm = np.linalg.norm(variant, axis=1) + 1e-6
    l2 = np.linalg.norm(diff, axis=1)
    rel = l2 / base_norm
    cosine = np.sum(base * variant, axis=1) / (base_norm * variant_norm)
    return {
        "mean_l2": float(np.mean(l2)),
        "median_l2": float(np.median(l2)),
        "p90_l2": float(np.percentile(l2, 90)),
        "mean_relative_l2": float(np.mean(rel)),
        "mean_cosine": float(np.mean(cosine)),
    }


def cross_image_baseline(features: np.ndarray, num_pairs: int = 1000, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    idx_a = rng.integers(0, n, size=num_pairs)
    idx_b = rng.integers(0, n, size=num_pairs)
    diff = features[idx_a] - features[idx_b]
    distances = np.linalg.norm(diff, axis=1)
    return {
        "mean_cross_image_l2": float(np.mean(distances)),
        "median_cross_image_l2": float(np.median(distances)),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    repo_root = Path(__file__).resolve().parent.parent
    bundle = load_dataset_bundle(repo_root, args.dataset, args.max_images)

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

    feature_store: Dict[str, torch.Tensor] = {}
    for key, prompt in PROMPT_VARIANTS.items():
        print(f"Collecting features for prompt variant: {key}")
        feats = collect_pooled_features(
            encoder,
            processor,
            bundle.image_paths,
            prompt,
            device,
            autocast_dtype,
            pooling=args.feature_pooling,
        )
        feature_store[key] = feats.clone().to(torch.float32)

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    base = feature_store["base"].numpy()
    summaries = {}
    for key, feats in feature_store.items():
        if key == "base":
            continue
        summaries[key] = summarize_differences(base, feats.numpy())
    summaries["cross_image"] = cross_image_baseline(base, seed=args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.dataset}_prompt_sensitivity.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "num_images": len(bundle.image_paths),
                "prompt_keys": list(PROMPT_VARIANTS.keys()),
                "summaries": summaries,
                "config": {
                    "model_name": args.model_name,
                    "feature_pooling": args.feature_pooling,
                },
            },
            f,
            indent=2,
        )
    print(f"Saved prompt sensitivity metrics to {output_path}")


if __name__ == "__main__":
    main()


