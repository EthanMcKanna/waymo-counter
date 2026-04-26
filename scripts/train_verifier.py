#!/usr/bin/env python3
"""
Train and benchmark the second-stage Waymo crop verifier.

The verifier consumes YOLO proposal crops and predicts whether the crop is a
true Waymo. It is trained from review_candidates.labeled.jsonl / verifier
dataset crops and reports candidate-level lift over the current YOLO-only
production behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


LABEL_TO_INDEX = {"not_waymo": 0, "waymo": 1}
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Example:
    crop_path: Path
    label: int
    split: str
    market: str
    domain: str
    candidate_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Waymo crop verifier.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arch",
        choices=["mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0"],
        default="mobilenet_v3_small",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-austin-recall", type=float, default=0.90)
    parser.add_argument("--min-overall-recall", type=float, default=0.80)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_examples(dataset: Path) -> list[Example]:
    manifest = dataset / "manifest.jsonl"
    examples: list[Example] = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            label = row.get("review_label")
            if label not in LABEL_TO_INDEX:
                continue
            crop_path = Path(row["crop_path"])
            if not crop_path.is_absolute():
                crop_path = Path.cwd() / crop_path
            examples.append(
                Example(
                    crop_path=crop_path,
                    label=LABEL_TO_INDEX[label],
                    split=row["split"],
                    market=str(row.get("market") or "unknown"),
                    domain=str(row.get("domain") or "unknown"),
                    candidate_id=str(row.get("candidate_id") or ""),
                )
            )
    return examples


class GreenAnnotationSuppress:
    """Remove bright lime annotation pixels from reviewed training crops."""

    def __call__(self, image: Image.Image) -> Image.Image:
        arr = np.asarray(image.convert("RGB")).copy()
        red = arr[:, :, 0]
        green = arr[:, :, 1]
        blue = arr[:, :, 2]
        mask = (green > 145) & (red < 120) & (blue < 140) & ((green - red) > 60)
        if mask.any():
            median = np.median(arr[~mask], axis=0) if (~mask).any() else np.array([127, 127, 127])
            arr[mask] = median.astype(np.uint8)
        return Image.fromarray(arr)


class CropDataset(Dataset):
    def __init__(self, examples: list[Example], transform):
        self.examples = examples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        with Image.open(example.crop_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, example.label, index


def make_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            GreenAnnotationSuppress(),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.03),
            transforms.RandomAutocontrast(p=0.25),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            GreenAnnotationSuppress(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def build_model(arch: str, pretrained: bool) -> nn.Module:
    if arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 2)
        return model
    if arch == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 2)
        return model
    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 2)
        return model
    raise ValueError(f"Unsupported architecture: {arch}")


def class_weights(examples: list[Example], device: torch.device) -> torch.Tensor:
    counts = Counter(example.label for example in examples)
    total = sum(counts.values())
    weights = [total / max(1, counts[index]) for index in range(2)]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_loader(
    examples: list[Example],
    transform,
    batch_size: int,
    workers: int,
    weighted: bool,
) -> DataLoader:
    dataset = CropDataset(examples, transform)
    sampler = None
    shuffle = not weighted
    if weighted:
        counts = Counter(example.label for example in examples)
        sample_weights = [1.0 / counts[example.label] for example in examples]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_pred = y_prob >= threshold
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


def grouped_metrics(
    examples: list[Example],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    attr: str,
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        groups[getattr(example, attr)].append(index)
    metrics = {}
    for group, indexes in groups.items():
        idx = np.array(indexes)
        metrics[group] = binary_metrics(y_true[idx], y_prob[idx], threshold)
        metrics[group]["support"] = int(len(indexes))
    return metrics


def choose_threshold(
    examples: list[Example],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_austin_recall: float,
    min_overall_recall: float,
) -> tuple[float, dict[str, float | int]]:
    thresholds = np.linspace(0.05, 0.95, 181)
    best_threshold = 0.5
    best_metrics = binary_metrics(y_true, y_prob, best_threshold)
    best_score = -math.inf
    markets = np.array([example.market for example in examples])
    austin_mask = markets == "austin"
    for threshold in thresholds:
        metrics = binary_metrics(y_true, y_prob, float(threshold))
        austin_metrics = (
            binary_metrics(y_true[austin_mask], y_prob[austin_mask], float(threshold))
            if austin_mask.any()
            else {"recall": 1.0}
        )
        meets_recall = (
            metrics["recall"] >= min_overall_recall
            and austin_metrics["recall"] >= min_austin_recall
        )
        score = metrics["f1"] + (0.02 * metrics["precision"])
        if not meets_recall:
            score -= 1.0
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics
            best_metrics["austin_recall"] = austin_metrics["recall"]
    return best_threshold, best_metrics


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, examples: list[Example], device: torch.device):
    model.eval()
    probs = np.zeros(len(examples), dtype=np.float32)
    y_true = np.zeros(len(examples), dtype=np.int64)
    for images, labels, indexes in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        batch_probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        idx = indexes.numpy()
        probs[idx] = batch_probs
        y_true[idx] = labels.numpy()
    return y_true, probs


def evaluate_split(
    model: nn.Module,
    examples: list[Example],
    transform,
    args: argparse.Namespace,
    device: torch.device,
    threshold: float,
) -> dict[str, Any]:
    loader = make_loader(examples, transform, args.batch_size, args.num_workers, weighted=False)
    y_true, y_prob = predict(model, loader, examples, device)
    return {
        "threshold": threshold,
        "overall": binary_metrics(y_true, y_prob, threshold),
        "by_market": grouped_metrics(examples, y_true, y_prob, threshold, "market"),
        "by_domain": grouped_metrics(examples, y_true, y_prob, threshold, "domain"),
        "baseline_yolo_only": binary_metrics(y_true, y_prob, 0.0),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    examples = load_examples(args.dataset)
    splits = {split: [example for example in examples if example.split == split] for split in ("train", "val", "test")}
    train_transform, eval_transform = make_transforms(args.image_size)

    model = build_model(args.arch, pretrained=not args.no_pretrained).to(device)
    train_loader = make_loader(
        splits["train"],
        train_transform,
        args.batch_size,
        args.num_workers,
        weighted=True,
    )
    val_loader = make_loader(splits["val"], eval_transform, args.batch_size, args.num_workers, weighted=False)
    criterion = nn.CrossEntropyLoss(weight=class_weights(splits["train"], device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_state = None
    best_val_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * labels.numel()
            total_items += labels.numel()
        scheduler.step()

        y_true, y_prob = predict(model, val_loader, splits["val"], device)
        threshold, val_metrics = choose_threshold(
            splits["val"],
            y_true,
            y_prob,
            args.min_austin_recall,
            args.min_overall_recall,
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, total_items),
            "val_threshold": threshold,
            "val_metrics": val_metrics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = float(val_metrics["f1"])
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true_val, y_prob_val = predict(model, val_loader, splits["val"], device)
    threshold, val_metrics = choose_threshold(
        splits["val"],
        y_true_val,
        y_prob_val,
        args.min_austin_recall,
        args.min_overall_recall,
    )
    val_eval = evaluate_split(model, splits["val"], eval_transform, args, device, threshold)
    test_eval = evaluate_split(model, splits["test"], eval_transform, args, device, threshold)

    torch.save(model.state_dict(), args.output / "verifier_state_dict.pt")
    model.eval()
    example = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    traced = torch.jit.trace(model, example)
    traced.save(str(args.output / "verifier.torchscript.pt"))

    summary = {
        "arch": args.arch,
        "best_epoch": best_epoch,
        "image_size": args.image_size,
        "threshold": threshold,
        "dataset": str(args.dataset),
        "split_counts": {split: Counter(INDEX_TO_LABEL[e.label] for e in rows) for split, rows in splits.items()},
        "val": val_eval,
        "test": test_eval,
        "history": history,
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    }
    summary = json.loads(json.dumps(summary, default=dict))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    shutil.copy2(args.dataset / "summary.json", args.output / "dataset_summary.json")
    print("FINAL_SUMMARY", json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
