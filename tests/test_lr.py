import numpy as np
import polars as pl

from meds_interp import knn_raw_code as knn
from meds_interp import meds_logistic_regression as lr


def test_lr():
    train_df = pl.DataFrame({"embeddings": [[1, 1, 0], [1, 2, 1], [2, 0, -2]], "label": [0, 1, 0]})
    test_df = pl.DataFrame({"embeddings": [[1, 2, 1], [1, 0, 0], [1, 0, 1]], "label": [1, 0, 1]})
    probabilities = lr.fit_logistic_regression(train_df, test_df, 1.0)
    test_df = test_df.with_columns(pl.Series("probabilities", probabilities))
    lr.score_labels(test_df)


def test_knn():
    train_df = pl.DataFrame({"embeddings": [[1, 1, 0], [1, 2, 1], [2, 0, -2]], "label": [0, 1, 0]})
    test_df = pl.DataFrame({"embeddings": [[1, 2, 1], [1, 0, 0], [1, 0, 1]], "label": [1, 0, 1]})
    probabilities = lr.fit_logistic_regression(train_df, test_df, 1.0)
    test_df = test_df.with_columns(pl.Series("probabilities", probabilities))
    lr.score_labels(test_df)


def test_dual_modality_knn():
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

    test1 = knn.DualFaissKNNClassifier(
        modalities=modalities, modality_weights=modality_weights, n_neighbors=2
    )
    test1.fit_preprocess(train_df_2)
    test1.transform_preprocess(train_df_2)
    test1.fit(train_df_2, label)
    x = test1.predict(train_df_2)
    print(x)
    assert x.shape == label.shape
    preprocess2 = knn.Preprocess_Type.NORM_AFTER_CONCAT
    test2 = knn.DualFaissKNNClassifier(
        modalities=modalities, modality_weights=modality_weights, preprocess=preprocess2
    )
    test2.fit_preprocess(train_df_2)
    test2.transform_preprocess(train_df_2)

    preprocess3 = knn.Preprocess_Type.NORM_SEPERATLY
    test3 = knn.DualFaissKNNClassifier(
        modalities=modalities, modality_weights=modality_weights, preprocess=preprocess3
    )
    test3.fit_preprocess(train_df_2)
    test3.transform_preprocess(train_df_2)
