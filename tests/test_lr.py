from meds_interp import meds_logistic_regression as lr
import sklearn

import polars as pl


def test_lr():
    df = pl.DataFrame({"embeddings": [[1,1,1], [1,2,1], [2,0,-2]], "labels": [0, 1, 0]})
    output = lr.fit_logistic_regression(df, 1.0)
    print("test started!")
    expected = pl.Series([0, 1, 0])
    