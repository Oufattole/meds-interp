from pathlib import Path

import polars as pl
from hydra import compose, initialize

from meds_interp import knn_raw_code as knn
from meds_interp import meds_logistic_regression as lr


def test_lr():
    train_df = pl.DataFrame({"embeddings": [[1, 1, 0], [1, 2, 1], [2, 0, -2]], "label": [0, 1, 0]})
    test_df = pl.DataFrame({"embeddings": [[1, 2, 1], [1, 0, 0], [1, 0, 1]], "label": [1, 0, 1]})
    probabilities = lr.fit_logistic_regression(train_df, test_df, 1.0)
    test_df = test_df.with_columns(pl.Series(probabilities).alias("probabilities"))
    lr.score_labels(test_df)


def test_knn_model():
    train_df_2 = pl.DataFrame(
        {
            "modality_1": [[1, 1, 0], [1, 2, 1], [2, 0, -2], [0, 1, 2]],
            "modality_2": [[1, 0, 0], [0, 2, 1], [4, 1, 1], [3, 1, 2]],
            "label": [0, 1, 0, 0],
        }
    )

    modalities = ["modality_1", "modality_2"]
    modality_weights = [4, 1]

    test1 = knn.KNN_Model(modalities=modalities, modality_weights=modality_weights, n_neighbors=2)
    test1.fit(train_df_2)
    x1 = test1.predict(train_df_2)
    assert x1.shape == train_df_2.get_column("label").shape

    preprocess2 = knn.Preprocess_Type.NORM_AFTER_CONCAT
    test2 = knn.KNN_Model(modalities=modalities, modality_weights=modality_weights, preprocess=preprocess2)
    test2.fit(train_df_2)
    x2 = test2.predict(train_df_2)
    assert x2.shape == train_df_2.get_column("label").shape

    preprocess3 = knn.Preprocess_Type.NORM_SEPERATLY
    test3 = knn.KNN_Model(modalities=modalities, modality_weights=modality_weights, preprocess=preprocess3)
    test3.fit(train_df_2)
    x3 = test3.predict(train_df_2)
    assert x3.shape == train_df_2.get_column("label").shape


def test_knn_tuning(tmp_path):
    train_df = pl.DataFrame(
        {
            "modality_1": [[1, 1, 0], [1, 2, 1], [2, 0, -2], [0, 1, 2]],
            "modality_2": [[1, 0, 0], [0, 2, 1], [4, 1, 1], [3, 1, 2]],
            "modality_3": [1, 2, 3, 4],
            "label": [0, 0, 1, 1],
        }
    )
    val_df = pl.DataFrame(
        {
            "modality_1": [[3, 5, 2], [4, 1, 3], [6, 2, 0], [7, 1, 1]],
            "modality_2": [[2, 3, 1], [5, 0, 4], [1, 6, 2], [3, 2, 5]],
            "modality_3": [2, 6, 1, 1],
            "label": [0, 1, 0, 1],
        }
    )
    test_df = pl.DataFrame(
        {
            "modality_1": [[-2, 3, -1], [0, -1, 2], [-3, 1, -2], [1, -2, 3]],
            "modality_2": [[3, -1, 0], [-2, 2, -3], [1, -3, 2], [0, 3, -1]],
            "modality_3": [1, 4, 2, 6],
            "label": [0, 0, 1, 1],
        }
    )
    train_df.write_parquet(Path(tmp_path) / "train.parquet")
    val_df.write_parquet(Path(tmp_path) / "val.parquet")
    test_df.write_parquet(Path(tmp_path) / "test.parquet")

    # import pdb; pdb.set_trace()

    test_config = {
        "modalities": ["modality_1", "modality_2", "modality_3"],
        "+weights.modality_1": 1,
        "+weights.modality_2": 1,
        "+weights.modality_3": 1,
        "input_path": tmp_path,
    }

    with initialize(version_base=None, config_path="../src/meds_interp/configs/"):  # path to config.yaml
        overrides = [f"{k}={v}" for k, v in test_config.items()]
        cfg = compose(config_name="knn", overrides=overrides)  # config.yaml
    knn.main(cfg)
