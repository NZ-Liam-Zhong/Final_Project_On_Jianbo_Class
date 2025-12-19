#!/usr/bin/env python
"""Rank text-only sensitivity of OpenVLA features w.r.t. noun/verb prompts."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import spacy
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# Reuse the same base prompt as other scripts for comparability
BASE_PROMPT = "In: What action should the robot take to push the trees away from the wall?\nOut:"
DEFAULT_NOUN_TEMPLATE = "In: Describe the {word} in the scene.\nOut:"
DEFAULT_VERB_TEMPLATE = "In: Please {word} the object carefully.\nOut:"

HF_CACHE_DIR = Path("/mnt/disk1/ilykyleliam/public")


@dataclass
class DatasetBundle:
    label: str
    image_paths: List[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text sensitivity ranking for OpenVLA features")
    parser.add_argument(
        "--dataset",
        choices=["lego_val", "fern", "coco_val"],
        default="lego_val",
        help="Image dataset used to anchor visual content",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=8,
        help="Maximum number of images to sample for each prompt (smaller = faster)",
    )
    parser.add_argument(
        "--coco-root",
        type=Path,
        default=Path("/mnt/disk1/ilykyleliam/public/datasets/coco2017"),
        help="COCO directory used to mine candidate nouns/verbs",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=200,
        help="Maximum number of nouns/verbs per category to evaluate",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=25,
        help="Minimum number of COCO images a word must appear in to be considered",
    )
    parser.add_argument(
        "--word-type",
        choices=["nouns", "verbs", "both"],
        default="both",
        help="Which word categories to include in the ranking",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/text_sensitivity"),
        help="Directory to save the ranked list",
    )
    parser.add_argument("--model-name", default="openvla/openvla-7b", help="Hugging Face model identifier")
    parser.add_argument(
        "--feature-pooling",
        choices=["mean", "mean_std"],
        default="mean_std",
        help="Pooling strategy for hidden states",
    )
    parser.add_argument("--base-prompt", default=BASE_PROMPT, help="Reference prompt for baseline features")
    parser.add_argument(
        "--noun-template",
        default=DEFAULT_NOUN_TEMPLATE,
        help="Prompt template for noun sensitivity (must contain {word})",
    )
    parser.add_argument(
        "--verb-template",
        default=DEFAULT_VERB_TEMPLATE,
        help="Prompt template for verb sensitivity (must contain {word})",
    )
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_hf_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    import os

    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir))


def get_device_and_precision() -> Tuple[torch.device, torch.dtype, torch.dtype | None]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return device, torch.float32, None
    bf16_supported = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    if bf16_supported:
        return device, torch.bfloat16, torch.bfloat16
    return device, torch.float16, torch.float16


def _resolve_lego_frame_path(base_dir: Path, file_path: str) -> Path:
    rel = file_path.replace("./", "")
    candidate = base_dir / f"{rel}.png" if not rel.endswith(".png") else base_dir / rel
    return candidate.resolve()


def load_lego_val_dataset(repo_root: Path, max_images: int | None) -> DatasetBundle:
    lego_dir = repo_root / "lego"
    val_dir = lego_dir / "val"
    meta_path = lego_dir / "transforms_val.json"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    frames = meta["frames"]
    if max_images is not None:
        frames = frames[:max_images]
    image_paths: List[Path] = []
    for frame in frames:
        resolved = _resolve_lego_frame_path(lego_dir, frame["file_path"])
        if not resolved.exists():
            resolved = val_dir / Path(frame["file_path"]).name
        if resolved.exists():
            image_paths.append(resolved)
    return DatasetBundle(label="lego_val", image_paths=image_paths)


def load_fern_dataset(repo_root: Path, max_images: int | None) -> DatasetBundle:
    fern_dir = repo_root / "fern"
    image_dir = fern_dir / "images"
    image_paths = sorted(image_dir.glob("*.JPG"))
    if max_images is not None:
        image_paths = image_paths[:max_images]
    return DatasetBundle(label="fern", image_paths=image_paths)


def load_coco_val_dataset(coco_root: Path, max_images: int | None) -> DatasetBundle:
    image_dir = coco_root / "val2017"
    if not image_dir.exists():
        raise FileNotFoundError(f"COCO val images not found at {image_dir}")
    image_paths = sorted(image_dir.glob("*.jpg"))
    if max_images is not None:
        image_paths = image_paths[:max_images]
    return DatasetBundle(label="coco_val", image_paths=image_paths)


def load_coco_val_dataset(coco_root: Path, max_images: int | None) -> DatasetBundle:
    image_dir = coco_root / "val2017"
    if not image_dir.exists():
        raise FileNotFoundError(f"COCO val images not found at {image_dir}")
    image_paths = sorted(image_dir.glob("*.jpg"))
    if max_images is not None:
        image_paths = image_paths[:max_images]
    return DatasetBundle(label="coco_val", image_paths=image_paths)


def load_captions(annotation_path: Path) -> Dict[int, List[str]]:
    with annotation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    id_to_captions: Dict[int, List[str]] = {}
    for ann in data["annotations"]:
        id_to_captions.setdefault(ann["image_id"], []).append(ann["caption"])
    return id_to_captions


def extract_ranked_words(
    coco_root: Path,
    min_count: int,
) -> Tuple[List[str], List[str]]:
    captions_path = coco_root / "annotations" / "captions_val2017.json"
    if not captions_path.exists():
        raise FileNotFoundError(f"Missing COCO captions file: {captions_path}")
    id_to_captions = load_captions(captions_path)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    noun_counter: Counter[str] = Counter()
    verb_counter: Counter[str] = Counter()
    for captions in id_to_captions.values():
        doc = nlp(" ".join(captions))
        noun_set = set()
        verb_set = set()
        for token in doc:
            if not token.text.isalpha() or token.is_stop:
                continue
            lemma = token.lemma_.lower()
            if token.pos_ in {"NOUN", "PROPN"}:
                noun_set.add(lemma)
            elif token.pos_ == "VERB":
                verb_set.add(lemma)
        noun_counter.update(noun_set)
        verb_counter.update(verb_set)

    noun_words = [word for word, count in noun_counter.most_common() if count >= min_count]
    verb_words = [word for word, count in verb_counter.most_common() if count >= min_count]
    return noun_words, verb_words


def collect_pooled_features(
    model: AutoModelForVision2Seq,
    processor: AutoProcessor,
    image_paths: Sequence[Path],
    prompt: str,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    pooling: str,
) -> torch.Tensor:
    from contextlib import nullcontext

    pooled: List[torch.Tensor] = []
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
            hidden = outputs.hidden_states[-1][0].to(torch.float32).cpu()
            if pooling == "mean":
                pooled.append(hidden.mean(dim=0))
            elif pooling == "mean_std":
                mean = hidden.mean(dim=0)
                std = hidden.std(dim=0, unbiased=False)
                pooled.append(torch.cat([mean, std], dim=0))
            else:
                raise ValueError(f"Unsupported pooling strategy: {pooling}")
    return torch.stack(pooled)


def summarize_difference(base: np.ndarray, variant: np.ndarray) -> Dict[str, float]:
    diff = variant - base
    base_norm = np.linalg.norm(base, axis=1, keepdims=True).clip(min=1e-6)
    variant_norm = np.linalg.norm(variant, axis=1, keepdims=True).clip(min=1e-6)
    l2 = np.linalg.norm(diff, axis=1)
    rel = l2 / base_norm.squeeze(1)
    cosine = np.sum(base * variant, axis=1) / (base_norm.squeeze(1) * variant_norm.squeeze(1))
    return {
        "mean_l2": float(np.mean(l2)),
        "median_l2": float(np.median(l2)),
        "p90_l2": float(np.percentile(l2, 90)),
        "mean_relative_l2": float(np.mean(rel)),
        "mean_cosine": float(np.mean(cosine)),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_hf_cache(HF_CACHE_DIR)
    repo_root = Path(__file__).resolve().parent.parent

    if args.dataset == "lego_val":
        bundle = load_lego_val_dataset(repo_root, args.max_images)
    elif args.dataset == "fern":
        bundle = load_fern_dataset(repo_root, args.max_images)
    elif args.dataset == "coco_val":
        bundle = load_coco_val_dataset(args.coco_root, args.max_images)
    else:
        raise ValueError(f"Unsupported dataset {args.dataset}")
    if len(bundle.image_paths) == 0:
        raise RuntimeError(f"No images found for dataset {args.dataset}")

    noun_words, verb_words = extract_ranked_words(args.coco_root, args.min_count)

    if args.word_type in {"nouns", "both"}:
        noun_words = noun_words[: args.max_words]
    else:
        noun_words = []
    if args.word_type in {"verbs", "both"}:
        verb_words = verb_words[: args.max_words]
    else:
        verb_words = []

    device, model_dtype, autocast_dtype = get_device_and_precision()
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()

    base_features = collect_pooled_features(
        model,
        processor,
        bundle.image_paths,
        args.base_prompt,
        device,
        autocast_dtype,
        pooling=args.feature_pooling,
    ).numpy()

    rankings: List[dict] = []

    def process_words(words: List[str], template: str, word_type: str) -> None:
        for word in words:
            prompt = template.format(word=word)
            variant = collect_pooled_features(
                model,
                processor,
                bundle.image_paths,
                prompt,
                device,
                autocast_dtype,
                pooling=args.feature_pooling,
            ).numpy()
            metrics = summarize_difference(base_features, variant)
            rankings.append(
                {
                    "word": word,
                    "word_type": word_type,
                    **metrics,
                }
            )

    process_words(noun_words, args.noun_template, "noun")
    process_words(verb_words, args.verb_template, "verb")

    rankings.sort(key=lambda x: x["mean_l2"], reverse=True)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{bundle.label}_text_sensitivity_rankings.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "rankings": rankings,
                "config": {
                    "dataset": args.dataset,
                    "num_images": len(bundle.image_paths),
                    "max_words_per_type": args.max_words,
                    "min_count": args.min_count,
                    "word_type": args.word_type,
                    "feature_pooling": args.feature_pooling,
                    "base_prompt": args.base_prompt,
                    "noun_template": args.noun_template,
                    "verb_template": args.verb_template,
                },
            },
            f,
            indent=2,
        )
    print(f"Saved text sensitivity rankings to {output_path}")


if __name__ == "__main__":
    main()


