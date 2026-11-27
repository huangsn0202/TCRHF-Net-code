"""Run one TCRHF seed; evaluate only the final epoch (use main_1.py for the paper)."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .train_utils import ExperimentConfig, run_experiment
except ImportError:
    from train_utils import ExperimentConfig, run_experiment


SCRIPT_DIR = Path(__file__).resolve().parent


def _relative(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (SCRIPT_DIR / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default="../../../data/feature/goosipcop")
    parser.add_argument("--split-file", default="../../../data/goosipcop/dataset_split.xlsx")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/tcrhf_paper_final/seed_42_final.pt",
    )
    parser.add_argument(
        "--result-dir", default="results/tcrhf_paper_final/seed_42"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.0004)
    parser.add_argument("--relation-loss-weight", type=float, default=0.2)
    parser.add_argument("--aggregation-hidden-dim", type=int, default=256)
    # Layer-specific dropout rates. Input dropout defaults to zero because it
    # directly destroys values in the fixed precomputed T/I/C representations.
    parser.add_argument("--input-dropout", type=float, default=0.0)
    parser.add_argument("--view-dropout", type=float, default=0.2)
    parser.add_argument("--relation-dropout", type=float, default=0.2)
    parser.add_argument("--fusion-dropout", type=float, default=0.2)
    parser.add_argument("--classifier-dropout", type=float, default=0.2)
    parser.add_argument("--gradient-clip-norm", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig(
        feature_dir=_relative(args.feature_dir),
        split_file=_relative(args.split_file),
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        relation_loss_weight=args.relation_loss_weight,
        aggregation_hidden_dim=args.aggregation_hidden_dim,
        input_dropout=args.input_dropout,
        view_dropout=args.view_dropout,
        relation_dropout=args.relation_dropout,
        fusion_dropout=args.fusion_dropout,
        classifier_dropout=args.classifier_dropout,
        gradient_clip_norm=args.gradient_clip_norm,
        device=args.device,
    )
    print(
        "Dropout settings: "
        f"input={config.input_dropout:.2f} "
        f"view={config.view_dropout:.2f} "
        f"relation={config.relation_dropout:.2f} "
        f"fusion={config.fusion_dropout:.2f} "
        f"classifier={config.classifier_dropout:.2f}"
    )
    result = run_experiment(
        config,
        seed=args.seed,
        checkpoint_path=_relative(args.checkpoint),
        result_dir=_relative(args.result_dir),
    )
    metrics = result["final_test_metrics"]
    print(
        f"Final test epoch: {result['final_epoch']} | "
        f"Accuracy={metrics['accuracy']:.4f} Precision={metrics['precision']:.4f} "
        f"Recall={metrics['recall']:.4f} F1={metrics['f1']:.4f}"
    )
    print(
        "Content-relation diagnostics: "
        f"distance={metrics['content_relation_distance_mean']:.4f} "
        f"cosine={metrics['content_relation_cosine_mean']:+.4f}"
    )
    print(f"Parameters: {result['parameter_count']:,}")
    print(f"Checkpoint: {result['checkpoint']}")
    print(f"Test diagnostics: {result['diagnostics']}")


if __name__ == "__main__":
    main()
