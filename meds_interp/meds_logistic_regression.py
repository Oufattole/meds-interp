import polars as pl
from sklearn.linear_model import LogisticRegression
import numpy as np

def fit_logistic_regression(df: pl.DataFrame, c) -> pl.Series:
    """
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

    Args:
        df (pl.DataFrame): input data, column `embeddings` has embeddings that you feed to logistic regression
    Returns:
        pl.Series: labels from logistic regression model
    """
    # embeddings = df.get_column("embeddings")
    # labels = df.get_column("labels")

    # classifier = LogisticRegression(C=c)
    # classifier.fit(embeddings, labels)
    # predicted_labels = classifier.predict(embeddings)
    # return predicted_labels
    embeddings = np.array(df['embeddings'].to_list())
    labels = np.array(df['labels'])
    
    classifier = LogisticRegression(C=c)
    classifier.fit(embeddings, labels)
    predicted_labels = classifier.predict(embeddings)

    return pl.Series(predicted_labels)
    