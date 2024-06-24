import polars as pl
from sklearn.linear_model import LogisticRegression
import numpy as np

def fit_logistic_regression(train_df: pl.DataFrame, test_df: pl.DataFrame, c) -> pl.Series:
    """Fits Logistic Regression model to training dataset and gets labels on test set.
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

    Args:
        train_df (pl.DataFrame): input data to fit logistic regression on, column
            `embeddings` has embeddings, column `label` has labels
        test_df (pl.DataFrame): data to generate predictions for. Has an `embeddings` column.
    Returns:
        pl.Series: labels from logistic regression model for the test_df embeddings
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

def score_labels(test_df: pl.DataFrame):
    """Evaluates predictions
    Args:
        test_df: has column "prediction" with model predictions and "label" with ground truth labels
    Returns:
        dictionary of different evaluation metrics.
    """
    return dict(
        auc=0,
        accuracy=0,
    )
