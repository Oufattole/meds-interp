import faiss
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def fit_kmeans(df):
    # TODO: make this work and add doctests
    ncentroids = 1024
    niter = 20
    verbose = True
    d = df.shape[1]
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
    kmeans.train(df)
    # kmeans.predict()


def k_means_score_labels(df):
    # Check for the necessary columns
    if "label" not in df or "cluster" not in df:
        raise ValueError("DataFrame must contain 'label' and 'cluster' columns.")

    # Filter NaN values
    df_filtered = df.dropna(subset=["label", "cluster"])

    nmi_score = normalized_mutual_info_score(df_filtered["label"], df_filtered["cluster"])

    contingency_table = pd.crosstab(df_filtered["cluster"], df_filtered["label"])
    total_purity = sum(contingency_table.max(axis=1)) / df_filtered.shape[0]

    ari_score = adjusted_rand_score(df_filtered["label"], df_filtered["cluster"])

    return dict(NMI=nmi_score, purity=total_purity, ari=ari_score)
