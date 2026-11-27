"""Run the paper protocol: ten fixed seeds, final-epoch evaluation, mean and sample SD."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

try:
    from .train_utils import ExperimentConfig, run_experiment, save_json
except ImportError:
    from train_utils import ExperimentConfig, run_experiment, save_json


SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_SEED = 42
SEED_MASTER_ENTROPY = 20260828
SEED_GENERATOR = "numpy.random.SeedSequence.spawn"
# The nine non-reference seeds were generated once with:
# np.random.SeedSequence(SEED_MASTER_ENTROPY).spawn(9), followed by one
# uint32 state value per child. They are fixed here and must not be selected
# according to experimental performance.
GENERATED_SEEDS = [
    144269559,
    74865246,
    1276543517,
    2351855581,
    181029087,
    723582294,
    3503254173,
    2754094657,
    3296097850,
]
DEFAULT_SEEDS = [REFERENCE_SEED, *GENERATED_SEEDS]
UINT32_MAX = 2**32 - 1
SUMMARY_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "class_0_precision",
    "class_0_recall",
    "class_0_f1",
    "class_1_precision",
    "class_1_recall",
    "class_1_f1",
    "content_relation_distance_mean",
    "content_relation_cosine_mean",
)


def _relative(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (SCRIPT_DIR / path).resolve()


def _parse_seeds(value: str) -> List[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(seeds) != 10 or len(set(seeds)) != 10:
        raise argparse.ArgumentTypeError("Exactly 10 distinct seeds are required.")
    if any(seed < 0 or seed > UINT32_MAX for seed in seeds):
        raise argparse.ArgumentTypeError(
            f"Every seed must be in the uint32 range [0, {UINT32_MAX}]."
        )
    return seeds


def _seed_protocol(seeds: List[int]) -> dict:
    uses_default = list(seeds) == DEFAULT_SEEDS
    return {
        "name": (
            "fixed_reference_plus_seedsequence_v1"
            if uses_default
            else "user_supplied_ten_seed_set"
        ),
        "uses_default_protocol": uses_default,
        "reference_seed": REFERENCE_SEED if uses_default else None,
        "master_entropy": SEED_MASTER_ENTROPY if uses_default else None,
        "generator": SEED_GENERATOR if uses_default else None,
        "seed_count": len(seeds),
        "ordered_seeds": list(seeds),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default="../../../data/feature/goosipcop")
    parser.add_argument("--split-file", default="../../../data/goosipcop/dataset_split.xlsx")
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/tcrhf_paper_ten_seed_final",
    )
    parser.add_argument(
        "--result-dir", default="results/tcrhf_paper_ten_seed_final"
    )
    parser.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=4e-4)
    parser.add_argument("--relation-loss-weight", type=float, default=0.20)
    parser.add_argument("--aggregation-hidden-dim", type=int, default=256)
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
    checkpoint_dir = _relative(args.checkpoint_dir)
    result_dir = _relative(args.result_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    seed_protocol = _seed_protocol(args.seeds)
    save_json(seed_protocol, result_dir / "seed_manifest.json")
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
    print(f"Seed protocol: {seed_protocol['name']}")
    print(f"Ordered seeds: {seed_protocol['ordered_seeds']}")

    all_epochs = []
    final_rows = []
    for index, seed in enumerate(args.seeds):
        print(f"\n===== Seed {seed} ({index + 1}/{len(args.seeds)}) =====")
        result = run_experiment(
            config,
            seed=seed,
            checkpoint_path=checkpoint_dir / f"seed_{seed}_final.pt",
            result_dir=result_dir / f"seed_{seed}",
            verify_feature_files=index == 0,
        )
        all_epochs.extend(result["history"])
        final_rows.append(
            {
                "seed_index": index + 1,
                "seed": seed,
                "final_epoch": result["final_epoch"],
                "parameter_count": result["parameter_count"],
                **result["final_test_metrics"],
                "checkpoint": result["checkpoint"],
                "diagnostics": result["diagnostics"],
            }
        )

    epoch_frame = pd.DataFrame(all_epochs)
    final_frame = pd.DataFrame(final_rows)
    epoch_frame.to_csv(
        result_dir / "all_epoch_metrics.csv", index=False, encoding="utf-8-sig"
    )
    final_frame.to_csv(
        result_dir / "final_metrics_by_seed.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        metric: {
            "mean": float(final_frame[metric].mean()),
            "std": float(final_frame[metric].std(ddof=1)),
        }
        for metric in SUMMARY_METRICS
    }
    save_json(
        {
            "architecture": "tcrhf_single_pair_content_relation_hierarchical_fusion",
            "seeds": args.seeds,
            "seed_protocol": seed_protocol,
            "epochs_per_seed": args.epochs,
            "selection_split": None,
            "selection_metric": "fixed_final_epoch",
            "config": config.to_dict(),
            "summary": summary,
        },
        result_dir / "ten_seed_summary.json",
    )
    pd.DataFrame(
        [
            {"metric": metric, "mean": values["mean"], "std": values["std"]}
            for metric, values in summary.items()
        ]
    ).to_csv(result_dir / "ten_seed_summary.csv", index=False, encoding="utf-8-sig")
    print("\n===== Ten-seed summary =====")
    for metric, values in summary.items():
        print(f"{metric}: {values['mean']:.4f} ± {values['std']:.4f}")
    print(f"Results: {result_dir}")
    print(f"Checkpoints: {checkpoint_dir}")


if __name__ == "__main__":
    main()
