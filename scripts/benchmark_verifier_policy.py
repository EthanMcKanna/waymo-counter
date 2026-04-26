#!/usr/bin/env python3
"""
Benchmark a TorchScript verifier with market-aware threshold policies.

This is intentionally separate from training so follow-up models can be compared
against the currently deployed policy without retraining or hand-editing JSON.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark verifier policy thresholds.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--austin-threshold", type=float, default=0.475)
    parser.add_argument("--non-austin-threshold", type=float, default=0.90)
    parser.add_argument("--min-recall", type=float, default=0.92)
    parser.add_argument("--min-austin-recall", type=float, default=0.90)
    return parser.parse_args()


def load_training_helpers():
    script_path = Path(__file__).resolve().parent / "train_verifier.py"
    spec = importlib.util.spec_from_file_location("train_verifier_helpers", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    true_pos = int(((y_true == 1) & y_pred).sum())
    false_pos = int(((y_true == 0) & y_pred).sum())
    false_neg = int(((y_true == 1) & (~y_pred)).sum())
    true_neg = int(((y_true == 0) & (~y_pred)).sum())
    precision = true_pos / (true_pos + false_pos) if true_pos + false_pos else 0.0
    recall = true_pos / (true_pos + false_neg) if true_pos + false_neg else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": true_pos,
        "fp": false_pos,
        "fn": false_neg,
        "tn": true_neg,
    }


def policy_thresholds(examples: list[Any], austin: float, non_austin: float) -> np.ndarray:
    return np.array(
        [austin if example.market == "austin" else non_austin for example in examples],
        dtype=np.float32,
    )


def policy_metrics(
    examples: list[Any],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    austin: float,
    non_austin: float,
) -> dict[str, Any]:
    thresholds = policy_thresholds(examples, austin, non_austin)
    y_pred = y_prob >= thresholds
    metrics: dict[str, Any] = binary_metrics(y_true, y_pred)
    by_market = {}
    for market in sorted({example.market for example in examples}):
        indexes = np.array([index for index, example in enumerate(examples) if example.market == market])
        by_market[market] = binary_metrics(y_true[indexes], y_pred[indexes])
        by_market[market]["support"] = int(len(indexes))
    metrics["by_market"] = by_market
    metrics["thresholds"] = {
        "austin_threshold": austin,
        "non_austin_threshold": non_austin,
    }
    return metrics


@torch.inference_mode()
def predict(model, loader, examples, device: torch.device):
    y_true = np.zeros(len(examples), dtype=np.int64)
    y_prob = np.zeros(len(examples), dtype=np.float32)
    model.eval()
    for images, labels, indexes in loader:
        logits = model(images.to(device, non_blocking=True))
        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        idx = indexes.numpy()
        y_true[idx] = labels.numpy()
        y_prob[idx] = probs
    return y_true, y_prob


def sweep_policy(
    examples: list[Any],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float,
    min_austin_recall: float,
) -> dict[str, Any]:
    markets = np.array([example.market for example in examples])
    austin_mask = markets == "austin"
    best: dict[str, Any] | None = None

    for austin in np.linspace(0.30, 0.80, 101):
        for non_austin in np.linspace(0.50, 0.99, 99):
            metrics = policy_metrics(examples, y_true, y_prob, float(austin), float(non_austin))
            if austin_mask.any():
                austin_pred = y_prob[austin_mask] >= float(austin)
                austin_metrics = binary_metrics(y_true[austin_mask], austin_pred)
                austin_recall = float(austin_metrics["recall"])
            else:
                austin_recall = 1.0

            if metrics["recall"] < min_recall or austin_recall < min_austin_recall:
                continue

            score = metrics["f1"] + (0.002 * metrics["precision"]) - (0.0005 * metrics["fp"])
            row = {
                "score": score,
                "austin_recall": austin_recall,
                **metrics,
            }
            if best is None or row["score"] > best["score"]:
                best = row

    if best is None:
        return {}
    return best


def main() -> None:
    args = parse_args()
    helpers = load_training_helpers()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    examples = helpers.load_examples(args.dataset)
    _, eval_transform = helpers.make_transforms(args.image_size)
    model = torch.jit.load(str(args.model), map_location=device)

    report: dict[str, Any] = {
        "model": str(args.model),
        "dataset": str(args.dataset),
        "fixed_policy": {
            "austin_threshold": args.austin_threshold,
            "non_austin_threshold": args.non_austin_threshold,
        },
        "splits": {},
    }
    for split in ("val", "test"):
        rows = [example for example in examples if example.split == split]
        loader = helpers.make_loader(
            rows,
            eval_transform,
            args.batch_size,
            args.num_workers,
            weighted=False,
        )
        y_true, y_prob = predict(model, loader, rows, device)
        report["splits"][split] = {
            "fixed_policy": policy_metrics(
                rows,
                y_true,
                y_prob,
                args.austin_threshold,
                args.non_austin_threshold,
            ),
            "best_policy": sweep_policy(
                rows,
                y_true,
                y_prob,
                args.min_recall,
                args.min_austin_recall,
            ),
            "yolo_only_baseline": helpers.binary_metrics(y_true, y_prob, 0.0),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
