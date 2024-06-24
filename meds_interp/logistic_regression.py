import polars as pl

def fit_logistic_regression(df: pl.DataFrame) -> pl.Series:
    """
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

    Args:
        df (pl.DataFrame): input data, column `embeddings` has embeddings that you feed to logistic regression
    Returns:
        pl.Series: labels from logistic regression model
    """
    return None
    