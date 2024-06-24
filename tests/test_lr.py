import polars as pl

from meds_interp import meds_logistic_regression as lr


def test_lr():
    train_df = pl.DataFrame({"embeddings": [[1, 1, 0], [1, 2, 1], [2, 0, -2]], "label": [0, 1, 0]})
    test_df = pl.DataFrame({"embeddings": [[1, 2, 1], [1, 0, 0], [1, 0, 1]], "label": [1, 0, 1]})
    predictions, probabilities = lr.fit_logistic_regression(train_df, test_df, 1.0)
    test_df = test_df.with_columns(
        [pl.Series("predictions", predictions), pl.Series("probabilities", probabilities)]
    )
    print(lr.score_labels(test_df))
    # output = lr.score_labels(test_df)
    # expected = dict(
    #     auc=0.7,
    #     accuracy=0.8,
    # )
