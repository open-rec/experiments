"""Run the Synerise evaluator with an indexed, label-equivalent dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", type=Path, required=True)
    args, official_args = parser.parse_known_args()
    sys.path.insert(0, str(args.evaluator.resolve()))

    from training_pipeline import data_module, train, train_runner
    from training_pipeline.dataset import BehavioralDataset
    from training_pipeline.target_calculators import (
        ChurnTargetCalculator,
        PropensityTargetCalculator,
    )

    class IndexedBehavioralDataset(BehavioralDataset):
        """Cache exact official targets once instead of filtering a frame per row."""

        def __init__(self, embeddings, client_ids, target_df, target_calculator):
            super().__init__(embeddings, client_ids, target_df, target_calculator)
            self.targets = np.zeros(
                (len(client_ids), target_calculator.target_dim), dtype=np.float32
            )
            client_to_row = {int(client_id): row for row, client_id in enumerate(client_ids)}
            if isinstance(target_calculator, ChurnTargetCalculator):
                self.targets[:, 0] = 1
                for client_id in target_df["client_id"].unique():
                    row = client_to_row.get(int(client_id))
                    if row is not None:
                        self.targets[row, 0] = 0
            elif isinstance(target_calculator, PropensityTargetCalculator):
                value_to_columns = {}
                for column, value in enumerate(target_calculator._propensity_targets):
                    value_to_columns.setdefault(value, []).append(column)
                pairs = target_df[
                    ["client_id", target_calculator._propensity_type]
                ].drop_duplicates()
                for client_id, value in pairs.itertuples(index=False, name=None):
                    row = client_to_row.get(int(client_id))
                    if row is not None:
                        for column in value_to_columns.get(value, ()):
                            self.targets[row, column] = 1
            else:
                raise TypeError(type(target_calculator))

            for row in np.linspace(0, len(client_ids) - 1, min(32, len(client_ids)), dtype=int):
                expected = target_calculator.compute_target(client_ids[row], target_df)
                np.testing.assert_array_equal(self.targets[row], expected)
            self.target_df = None

        def __getitem__(self, idx):
            return self.embeddings[idx], self.targets[idx]

    data_module.BehavioralDataset = IndexedBehavioralDataset
    official_pl = train_runner.pl
    official_trainer = official_pl.Trainer

    class EpochNotice(official_pl.Callback):
        def on_train_epoch_end(self, trainer, _model):
            print(f"completed epoch {trainer.current_epoch + 1}/{trainer.max_epochs}",
                  flush=True)

    def trainer_without_checkpoints(*trainer_args, **trainer_kwargs):
        # Checkpoints are not used by the organizer's score aggregator.
        trainer_kwargs.setdefault("enable_checkpointing", False)
        trainer_kwargs["callbacks"] = list(trainer_kwargs.get("callbacks") or []) + [
            EpochNotice()]
        return official_trainer(*trainer_args, **trainer_kwargs)

    train_runner.pl = SimpleNamespace(Trainer=trainer_without_checkpoints)
    params = train.get_parser().parse_args(official_args)
    if params.score_dir:
        Path(params.score_dir).mkdir(parents=True, exist_ok=True)
    train.main(params)


if __name__ == "__main__":
    main()
