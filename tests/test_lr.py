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
    modalities = ["embeddings"]
    # modalities
    test_df = pl.DataFrame({"embeddings": [[1, 2, 1], [1, 0, 0], [1, 0, 1]], "label": [1, 0, 1]})
    probabilities = lr.fit_logistic_regression(train_df, test_df, 1.0)
    test_df = test_df.with_columns(pl.Series("probabilities", probabilities))
    lr.score_labels(test_df)


def test_dual_modality_knn():
    train_df = pl.DataFrame(
        {
            "embeddings_0": [[1, 1, 0], [1, 2, 1], [2, 0, -2]],
            "embeddings_1": [[1, 0, 0], [0, 2, 1], [-2, 0, -5]],
            "label": [0, 1, 0]
        }
    )
    train_df_2 = pl.DataFrame(
    {
        "embeddings": ["embedding_1", "embedding_2"],
        "modality_1": [[[1, 1, 0], [1, 2, 1], [2, 0, -2]], [[0, 1, 2], [4, 0, 3], [1, 1, -1]]],
        "modality_2": [[[1, 0, 0], [0, 2, 1], [-2, 0, -5]], [[4, 1, 1], [3, 1, 2], [2, 1, 4]]]
    }
)

    modalities = ["modality_1", "modality_2"]
    modality_weights = [4, 1]
    print(train_df_2)
    assert False

    
