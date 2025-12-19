#!/usr/bin/env python
"""Prototype-based verb/noun alignment analysis on COCO captions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import spacy
import torch

COCO_IMAGES_SUBDIR = "val2017"
COCO_CAPTIONS_FILE = "annotations/captions_val2017.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COCO verb/noun prototype retrieval using OpenVLA features")
    parser.add_argument(
        "--coco-root",
        type=Path,
        default=Path("/mnt/disk1/ilykyleliam/public/datasets/coco2017"),
        help="Path to COCO2017 directory",
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=Path("experiments/coco_attribute_probe/val_openvla_features.pt"),
        help="Feature cache produced by coco_attribute_probe.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/coco_verb_noun_alignment"),
        help="Directory for metrics",
    )
    parser.add_argument("--top-nouns", type=int, default=20, help="Number of nouns to evaluate")
    parser.add_argument("--top-verbs", type=int, default=15, help="Number of verbs to evaluate")
    parser.add_argument("--min-count", type=int, default=25, help="Minimum image count for a word to be kept")
    parser.add_argument("--seed", type=int, default=17, help="Random seed for reproducibility")
    return parser.parse_args()


def load_captions(annotation_path: Path) -> Dict[int, List[str]]:
    with annotation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    id_to_captions: Dict[int, List[str]] = {}
    for ann in data["annotations"]:
        id_to_captions.setdefault(ann["image_id"], []).append(ann["caption"])
    return id_to_captions


def extract_pos_sets(
    captions: Sequence[str],
    nlp,
) -> Tuple[Set[str], Set[str]]:
    nouns: Set[str] = set()
    verbs: Set[str] = set()
    joined = " ".join(captions)
    doc = nlp(joined)
    for token in doc:
        if not token.text.isalpha():
            continue
        lemma = token.lemma_.lower()
        if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop:
            nouns.add(lemma)
        elif token.pos_ == "VERB" and not token.is_stop:
            verbs.add(lemma)
    return nouns, verbs


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    positives = y_true.sum()
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores)
    sorted_true = y_true[order]
    cumulative = np.cumsum(sorted_true)
    precision = cumulative / (np.arange(len(sorted_true)) + 1)
    ap = (precision * sorted_true).sum() / positives
    return float(ap)


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(-scores)[:k]
    return float(y_true[order].sum() / max(k, 1))


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    positives = y_true.sum()
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores)[:k]
    return float(y_true[order].sum() / positives)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not args.feature_cache.exists():
        raise FileNotFoundError(
            f"Feature cache {args.feature_cache} not found. Run scripts/coco_attribute_probe.py first."
        )
    payload = torch.load(args.feature_cache)
    features = payload["features"].clone().to(torch.float32)
    image_ids: List[int] = payload["image_ids"]

    captions = load_captions(args.coco_root / COCO_CAPTIONS_FILE)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    noun_counter: Counter[str] = Counter()
    verb_counter: Counter[str] = Counter()
    image_nouns: List[Set[str]] = []
    image_verbs: List[Set[str]] = []

    for image_id in image_ids:
        caps = captions.get(image_id, [])
        n_set, v_set = extract_pos_sets(caps, nlp)
        image_nouns.append(n_set)
        image_verbs.append(v_set)
        noun_counter.update(n_set)
        verb_counter.update(v_set)

    def select_words(counter: Counter[str], limit: int) -> List[str]:
        words = [word for word, count in counter.most_common() if count >= args.min_count]
        return words[:limit]

    selected_nouns = select_words(noun_counter, args.top_nouns)
    selected_verbs = select_words(verb_counter, args.top_verbs)

    if not selected_nouns or not selected_verbs:
        raise RuntimeError("Not enough nouns or verbs met the minimum count threshold.")

    feature_np = features.numpy()
    feature_norms = np.linalg.norm(feature_np, axis=1, keepdims=True).clip(min=1e-6)
    normalized_features = feature_np / feature_norms

    def evaluate_words(word_list: List[str], per_image_sets: List[Set[str]]) -> List[dict]:
        results = []
        for word in word_list:
            labels = np.array([1 if word in word_set else 0 for word_set in per_image_sets], dtype=np.float32)
            if labels.sum() == 0:
                continue
            mask = labels.astype(bool)
            prototype = normalized_features[mask].mean(axis=0)
            proto_norm = np.linalg.norm(prototype).clip(min=1e-6)
            prototype /= proto_norm
            scores = normalized_features @ prototype
            ap = average_precision(labels, scores)
            p_at_10 = precision_at_k(labels, scores, k=10)
            r_at_10 = recall_at_k(labels, scores, k=10)
            results.append(
                {
                    "word": word,
                    "positives": int(labels.sum()),
                    "average_precision": ap,
                    "precision_at_10": p_at_10,
                    "recall_at_10": r_at_10,
                }
            )
        return results

    noun_results = evaluate_words(selected_nouns, image_nouns)
    verb_results = evaluate_words(selected_verbs, image_verbs)

    def summarize(results: List[dict]) -> dict:
        aps = [r["average_precision"] for r in results if not np.isnan(r["average_precision"])]
        p10 = [r["precision_at_10"] for r in results]
        r10 = [r["recall_at_10"] for r in results if not np.isnan(r["recall_at_10"])]
        return {
            "mean_average_precision": float(np.mean(aps)) if aps else None,
            "mean_precision_at_10": float(np.mean(p10)) if p10 else None,
            "mean_recall_at_10": float(np.mean(r10)) if r10 else None,
            "num_words": len(results),
        }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "nouns": noun_results,
        "verbs": verb_results,
        "noun_summary": summarize(noun_results),
        "verb_summary": summarize(verb_results),
        "config": {
            "feature_cache": str(args.feature_cache),
            "top_nouns": args.top_nouns,
            "top_verbs": args.top_verbs,
            "min_count": args.min_count,
        },
    }
    metrics_path = output_dir / "verb_noun_alignment_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved verb/noun alignment metrics to {metrics_path}")


if __name__ == "__main__":
    main()


