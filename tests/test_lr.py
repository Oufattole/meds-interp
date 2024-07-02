import tempfile
from pathlib import Path

import numpy as np
import polars as pl
from hydra import compose, initialize

from meds_interp import knn_raw_code as knn
from meds_interp import meds_logistic_regression as lr


def test_lr():
    train_df = pl.DataFrame({"embeddings": [[1, 1, 0], [1, 2, 1], [2, 0, -2]], "label": [0, 1, 0]})
    test_df = pl.DataFrame({"embeddings": [[1, 2, 1], [1, 0, 0], [1, 0, 1]], "label": [1, 0, 1]})
    probabilities = lr.fit_logistic_regression(train_df, test_df, 1.0)
    test_df = test_df.with_columns(pl.Series("probabilities", probabilities))
    lr.score_labels(test_df)


def test_knn_model():
    # train_df = pl.DataFrame(
    #     {
    #         "modality_0": [[1, 1, 0], [1, 2, 1], [2, 0, -2]],
    #         "modality_1": [[1, 0, 0], [0, 2, 1], [-2, 0, -5]],
    #         "label": [0, 1, 0],
    #     }
    # )

    train_df_2 = pl.DataFrame(
        {
            "modality_1": [[1, 1, 0], [1, 2, 1], [2, 0, -2], [0, 1, 2]],
            "modality_2": [[1, 0, 0], [0, 2, 1], [4, 1, 1], [3, 1, 2]],
        }
    )

    label = np.array([0, 1, 0, 0])
    modalities = ["modality_1", "modality_2"]
    modality_weights = [4, 1]

    test1 = knn.KNN_Model(modalities=modalities, modality_weights=modality_weights, n_neighbors=2)
    test1.fit_preprocess(train_df_2)
    test1.transform_preprocess(train_df_2)
    test1.fit(train_df_2, label)
    x = test1.predict(train_df_2)
    print(x)
    assert x.shape == label.shape
    preprocess2 = knn.Preprocess_Type.NORM_AFTER_CONCAT
    test2 = knn.KNN_Model(modalities=modalities, modality_weights=modality_weights, preprocess=preprocess2)
    test2.fit_preprocess(train_df_2)
    test2.transform_preprocess(train_df_2)

    preprocess3 = knn.Preprocess_Type.NORM_SEPERATLY
    test3 = knn.KNN_Model(modalities=modalities, modality_weights=modality_weights, preprocess=preprocess3)
    test3.fit_preprocess(train_df_2)
    test3.transform_preprocess(train_df_2)


def test_knn_tuning():
    with tempfile.TemporaryDirectory() as d:
        # TODO make the dummy data in d
        train_df = pl.DataFrame(
            {
                "modality_1": [[1, 1, 0], [1, 2, 1], [2, 0, -2], [0, 1, 2]],
                "modality_2": [[1, 0, 0], [0, 2, 1], [4, 1, 1], [3, 1, 2]],
            }
        )
        train_df.write_parquet(Path(d) / "train.parquet")
        # val_df.write_parquet(Path(d) / "val.parquet")
        # test_df.write_parquet(Path(d) / "test.parquet")

        test_config = dict(
            modalities=["m1", "m2"],
            modality_weights=[1, 1],
            input_path=d,
        )

        with initialize(version_base=None, config_path="../src/meds_interp/configs/"):  # path to config.yaml
            overrides = [f"{k}={v}" for k, v in test_config.items()]
            cfg = compose(config_name="knn", overrides=overrides)  # config.yaml
        knn.main(cfg)
