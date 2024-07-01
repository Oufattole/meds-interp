import numpy as np
import polars as pl
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


def fit_knn(train_df: pl.DataFrame, test_df: pl.DataFrame, c) -> pl.Series:
    """Fits Logistic Regression model to training dataset and gets labels on test set.
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

    Args:
        train_df (pl.DataFrame): input data to fit logistic regression on, column
            `embeddings` has embeddings, column `label` has labels
        test_df (pl.DataFrame): data to generate predictions for. Has an `embeddings` column.
    Returns:
        pl.Series: labels from logistic regression model for the test_df embeddings
    """

    train_embeddings = np.asarray(train_df["embeddings"].to_list())
    labels = train_df["label"].to_numpy()

    test_embeddings = np.asarray(test_df["embeddings"].to_list())

    classifier = LogisticRegression(C=c)
    classifier.fit(train_embeddings, labels)
    predicted_probabilities = classifier.predict_proba(test_embeddings)

    return pl.Series(predicted_probabilities)


def score_labels(test_df: pl.DataFrame):
    """Evaluates predictions
    Args:
        test_df: has column "prediction" with model predictions and "label" with ground truth labels
    Returns:
        dictionary of different evaluation metrics.
    """
    true_labels = test_df["label"]

    predicted_labels = test_df.get_column("probabilities").list.arg_max()
    predicted_probs = test_df.get_column("probabilities").list.get(0)

    accuracy = metrics.accuracy_score(true_labels, predicted_labels)
    auc_score = metrics.roc_auc_score(true_labels, predicted_probs)
    return dict(
        auc=float(auc_score),
        accuracy=accuracy,
    )
