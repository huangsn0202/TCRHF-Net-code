"""Fixed-epoch training and final-epoch-only evaluation (paper section 4.3)."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

try:
    from .model_0 import create_dataloaders, split_statistics
    from .model_2 import RELATION_NAMES
    from .model_4 import ThreeViewRumorModel
    from .model_5 import JointLoss
except ImportError:
    from model_0 import create_dataloaders, split_statistics
    from model_2 import RELATION_NAMES
    from model_4 import ThreeViewRumorModel
    from model_5 import JointLoss


@dataclass(frozen=True)
class ExperimentConfig:
    feature_dir: Path
    split_file: Path
    epochs: int = 20
    batch_size: int = 32
    num_workers: int = 0
    learning_rate: float = 1e-4
    weight_decay: float = 4e-4
    input_dim: int = 1024
    feature_dim: int = 1024
    decision_dim: int = 256
    aggregation_hidden_dim: int = 256
    relation_loss_weight: float = 0.20
    input_dropout: float = 0.0
    view_dropout: float = 0.2
    relation_dropout: float = 0.2
    fusion_dropout: float = 0.2
    classifier_dropout: float = 0.2
    gradient_clip_norm: float = 0.0
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive.")
        if self.num_workers < 0 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid data-loader or optimizer hyperparameters.")
        if self.input_dim != 1024 or self.feature_dim != 1024:
            raise ValueError("The model requires 1024-dimensional T/I/C features.")
        if 4 * self.decision_dim != self.feature_dim:
            raise ValueError("decision_dim must be 256 when feature_dim is 1024.")
        if self.aggregation_hidden_dim <= 0:
            raise ValueError("aggregation_hidden_dim must be positive.")
        if self.gradient_clip_norm < 0:
            raise ValueError("gradient_clip_norm must be non-negative.")
        if self.relation_loss_weight < 0:
            raise ValueError("relation_loss_weight must be non-negative.")
        dropout_settings = {
            "input_dropout": self.input_dropout,
            "view_dropout": self.view_dropout,
            "relation_dropout": self.relation_dropout,
            "fusion_dropout": self.fusion_dropout,
            "classifier_dropout": self.classifier_dropout,
        }
        for name, value in dropout_settings.items():
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1), got {value}.")

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["feature_dir"] = str(self.feature_dir)
        payload["split_file"] = str(self.split_file)
        return payload


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def compute_metrics(labels: Iterable[int], predictions: Iterable[int]) -> Dict[str, float]:
    labels_array = np.asarray(list(labels), dtype=np.int64)
    prediction_array = np.asarray(list(predictions), dtype=np.int64)
    if labels_array.size == 0:
        raise ValueError("Cannot compute metrics for an empty dataset.")
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels_array,
        prediction_array,
        labels=[0, 1],
        average="macro",
        zero_division=0,
    )
    class_p, class_r, class_f1, support = precision_recall_fscore_support(
        labels_array,
        prediction_array,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels_array, prediction_array)),
        "precision": float(macro_p),
        "recall": float(macro_r),
        "f1": float(macro_f1),
        "class_0_precision": float(class_p[0]),
        "class_0_recall": float(class_r[0]),
        "class_0_f1": float(class_f1[0]),
        "class_0_support": int(support[0]),
        "class_1_precision": float(class_p[1]),
        "class_1_recall": float(class_r[1]),
        "class_1_f1": float(class_f1[1]),
        "class_1_support": int(support[1]),
    }


def _move_batch(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {
        "id": batch["id"],
        "text": batch["text"].to(device, non_blocking=True),
        "image": batch["image"].to(device, non_blocking=True),
        "caption": batch["caption"].to(device, non_blocking=True),
        "label": batch["label"].to(device, non_blocking=True),
    }


def _class_mean(values: np.ndarray, labels: np.ndarray, class_id: int) -> float:
    selected = values[labels == class_id]
    return float(selected.mean()) if selected.size else 0.0


def run_epoch(
    model: ThreeViewRumorModel,
    objective: JointLoss,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    gradient_clip_norm: float = 0.0,
    diagnostics_path: Optional[Path] = None,
) -> Dict[str, float]:
    training = optimizer is not None
    if training and diagnostics_path is not None:
        raise ValueError("Per-sample diagnostics are only exported during evaluation.")
    diagnostic_rows: List[Dict[str, object]] = []
    model.train(training)
    objective.train(training)
    labels: List[int] = []
    predictions: List[int] = []
    relation_predictions = {name: [] for name in RELATION_NAMES}
    diagnostics = {
        "content_relation_distance": [],
        "content_relation_cosine": [],
        "z_content_norm": [],
        "z_relation_norm": [],
        "difference_norm": [],
        "product_norm": [],
    }
    loss_totals: Dict[str, float] = {}
    sample_count = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["text"], batch["image"], batch["caption"])
            loss, parts = objective(outputs, batch["label"])
            if training:
                loss.backward()
                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()

            current_batch_size = int(batch["label"].size(0))
            sample_count += current_batch_size
            for key, value in parts.items():
                loss_totals[key] = loss_totals.get(key, 0.0) + float(
                    value.detach().cpu()
                ) * current_batch_size
            labels.extend(batch["label"].detach().cpu().tolist())
            predictions.extend(outputs["logits"].argmax(-1).detach().cpu().tolist())
            for name in RELATION_NAMES:
                relation_predictions[name].extend(
                    outputs[f"logits_{name}"].argmax(-1).detach().cpu().tolist()
                )
            diagnostic_tensors = {
                "content_relation_distance": outputs["content_relation_distance"],
                "content_relation_cosine": outputs["content_relation_cosine"],
                "z_content_norm": outputs["z_content"].norm(dim=-1),
                "z_relation_norm": outputs["z_relation"].norm(dim=-1),
                "difference_norm": outputs["content_relation_difference"].norm(dim=-1),
                "product_norm": outputs["content_relation_product"].norm(dim=-1),
            }
            for name, tensor in diagnostic_tensors.items():
                diagnostics[name].extend(tensor.detach().cpu().tolist())

            if diagnostics_path is not None:
                batch_labels = batch["label"].detach().cpu().tolist()
                batch_probabilities = outputs["probabilities"].detach().cpu()
                batch_predictions = batch_probabilities.argmax(-1).tolist()
                batch_diagnostics = {
                    name: tensor.detach().cpu().tolist()
                    for name, tensor in diagnostic_tensors.items()
                }
                auxiliary = {
                    name: outputs[f"logits_{name}"].softmax(-1).detach().cpu()
                    for name in RELATION_NAMES
                }
                for index, sample_id in enumerate(batch["id"]):
                    row = {
                        "id": str(sample_id),
                        "label": batch_labels[index],
                        "prediction": batch_predictions[index],
                        "rumor_probability": float(batch_probabilities[index, 1]),
                        **{name: values[index] for name, values in batch_diagnostics.items()},
                    }
                    for name, probabilities in auxiliary.items():
                        row[f"relation_{name}_prediction"] = int(probabilities[index].argmax())
                        row[f"relation_{name}_rumor_probability"] = float(probabilities[index, 1])
                    diagnostic_rows.append(row)

    if diagnostics_path is not None:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(diagnostic_rows).to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
    metrics = compute_metrics(labels, predictions)
    for name, values in relation_predictions.items():
        metrics.update(
            {
                f"relation_{name}_{key}": value
                for key, value in compute_metrics(labels, values).items()
            }
        )
    metrics.update(
        {f"loss_{key}": value / sample_count for key, value in loss_totals.items()}
    )
    label_array = np.asarray(labels, dtype=np.int64)
    for name, values in diagnostics.items():
        value_array = np.asarray(values, dtype=np.float64)
        metrics[f"{name}_mean"] = float(value_array.mean())
        metrics[f"{name}_class_0"] = _class_mean(value_array, label_array, 0)
        metrics[f"{name}_class_1"] = _class_mean(value_array, label_array, 1)
    return metrics



def save_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


def run_experiment(
    config: ExperimentConfig,
    seed: int,
    checkpoint_path: Path,
    result_dir: Path,
    verify_feature_files: bool = True,
) -> Dict[str, object]:
    set_global_seed(seed)
    device = resolve_device(config.device)
    train_loader, test_loader = create_dataloaders(
        batch_size=config.batch_size,
        seed=seed,
        num_workers=config.num_workers,
        feature_dir=config.feature_dir,
        split_file=config.split_file,
        verify_files=verify_feature_files,
    )
    model = ThreeViewRumorModel(
        input_dim=config.input_dim,
        feature_dim=config.feature_dim,
        decision_dim=config.decision_dim,
        aggregation_hidden_dim=config.aggregation_hidden_dim,
        input_dropout=config.input_dropout,
        view_dropout=config.view_dropout,
        relation_dropout=config.relation_dropout,
        fusion_dropout=config.fusion_dropout,
        classifier_dropout=config.classifier_dropout,
    ).to(device)
    objective = JointLoss(config.relation_loss_weight).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, object]] = []
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    data_statistics = split_statistics(config.split_file)
    event_audit = train_loader.dataset.event_split_audit

    # No test-set access, model selection or early stopping inside this loop.
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(
            model, objective, train_loader, device,
            optimizer=optimizer,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        row: Dict[str, object] = {"seed": seed, "epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        history.append(row)
        print(
            f"seed={seed:>4} epoch={epoch:02d}/{config.epochs} "
            f"train_loss={train_metrics['loss_total']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"train_f1={train_metrics['f1']:.4f}"
        )

    # Freeze the final state before inspecting any test predictions or labels.
    torch.save(
        {
            "architecture": "tcrhf_single_pair_content_relation_hierarchical_fusion",
            "seed": seed,
            "epoch": config.epochs,
            "selection_split": None,
            "selection_metric": "fixed_final_epoch",
            "parameter_count": parameter_count,
            "config": config.to_dict(),
            "split_statistics": data_statistics,
            "event_split_audit": event_audit,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    pd.DataFrame(history).to_csv(
        result_dir / "epoch_metrics.csv", index=False, encoding="utf-8-sig"
    )
    save_json(history, result_dir / "epoch_metrics.json")

    diagnostics_path = result_dir / "test_diagnostics.csv"
    final_test_metrics = run_epoch(
        model, objective, test_loader, device, diagnostics_path=diagnostics_path
    )
    summary = {
        "architecture": "tcrhf_single_pair_content_relation_hierarchical_fusion",
        "seed": seed,
        "final_epoch": config.epochs,
        "final_test_metrics": final_test_metrics,
        "selection_split": None,
        "selection_metric": "fixed_final_epoch",
        "parameter_count": parameter_count,
        "checkpoint": str(checkpoint_path),
        "diagnostics": str(diagnostics_path),
        "device": str(device),
        "config": config.to_dict(),
        "split_statistics": data_statistics,
        "event_split_audit": event_audit,
        "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda},
    }
    save_json(summary, result_dir / "final_result.json")
    return {"history": history, **summary}
