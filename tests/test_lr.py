from meds_interp import logistic_regression as lr
import polars as pl


def test_lr():
    df = pl.DataFrame({"embeddings": [[1,1,1], [1,2,1], [2,0,-2]], "label": [0, 1, 0]})
    output = lr.fit_logistic_regression(None)
    print("test started!")
    expected = pl.Series([0, 1, 0])
    