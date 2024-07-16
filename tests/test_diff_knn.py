from pathlib import Path

import polars as pl
from hydra import compose, initialize

from meds_interp import diff_knn


def test_knn_tuning(tmp_path):
    train_df = pl.DataFrame(
        {
            "modality_1": [[1, 1, 0], [1, 2, 1], [2, 0, -2], [0, 1, 2]],
            "modality_2": [[1, 0, 0], [0, 2, 1], [4, 1, 1], [3, 1, 2]],
            "label": [0, 0, 1, 1],
        }
    )
    val_df = pl.DataFrame(
        {
            "modality_1": [[3, 5, 2], [4, 1, 3], [6, 2, 0], [7, 1, 1]],
            "modality_2": [[2, 3, 1], [5, 0, 4], [1, 6, 2], [3, 2, 5]],
            "label": [0, 1, 0, 1],
        }
    )
    test_df = pl.DataFrame(
        {
            "modality_1": [[-2, 3, -1], [0, -1, 2], [-3, 1, -2], [1, -2, 3]],
            "modality_2": [[3, -1, 0], [-2, 2, -3], [1, -3, 2], [0, 3, -1]],
            "label": [0, 0, 1, 1],
        }
    )
    train_df.write_parquet(Path(tmp_path) / "train.parquet")
    val_df.write_parquet(Path(tmp_path) / "val.parquet")
    test_df.write_parquet(Path(tmp_path) / "test.parquet")

    # import pdb; pdb.set_trace()

    test_config = {
        "modalities": ["modality_1", "modality_2"],
        "+weights.modality_1": 1,
        "+weights.modality_2": 1,
        "input_path": tmp_path,
    }

    with initialize(version_base=None, config_path="../src/meds_interp/configs/"):  # path to config.yaml
        overrides = [f"{k}={v}" for k, v in test_config.items()]
        cfg = compose(config_name="knn", overrides=overrides)  # config.yaml
    diff_knn.main(cfg)
