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
            "label": [0, 1, 0],
        }
    )
    # extract arrays
    # knn test_embedding -> find neighbors in the training set -> average the labels of those neighbors
    #
    embedd_array = np.asarray(train_df["embeddings_0"].to_list())
    modalities = ["embeddings_0", "embeddings_1"]

    embeddings_0 = np.array(train_df["embeddings_0"].to_list(), dtype=np.float32)
    embeddings_1 = np.array(train_df["embeddings_1"].to_list(), dtype=np.float32)

    # Concatenate the embeddings along the feature axis
    X = np.hstack((embeddings_0, embeddings_1))
    y = np.array(train_df["label"].to_list(), dtype=np.float32)
    print(X)

    modalities = ["embeddings_0", "embeddings_1"]
    d = embeddings_0.shape[1]  # Dimensionality of each embedding, which is 3

    # Verify d and X.shape[1]

    test1 = knn.DualFaissKNNClassifier(modalities, d=d)

    # Fit the classifier on the NumPy arrays
    test1.fit(X, y)
