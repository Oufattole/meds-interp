import faiss
import pandas as pd
import polars as pl
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def kmeans(df: pd.DataFrame, n_clusters: int):
    kmeans = faiss.Kmeans(d=df.shape[1], k=n_clusters, niter=20, verbose=True)
    kmeans.train(df)

    _, I = kmeans.index.search(df, 1)
    I_flat = [i[0] for i in I]

    return pd.Series(I_flat)


def k_means_score_labels(df):
    # Check for the necessary columns
    if "label" not in df or "cluster" not in df:
        raise ValueError("DataFrame must contain 'label' and 'cluster' columns.")

    # Filter NaN values
    df_filtered = df.filter(pl.col("label").is_not_null() & pl.col("cluster").is_not_null())

    nmi_score = normalized_mutual_info_score(df_filtered["label"], df_filtered["cluster"])

    pandas_df_pl = df.to_pandas()
    contingency_table_pd = pd.crosstab(pandas_df_pl["cluster"], pandas_df_pl["label"])
    contingency_table = pl.from_pandas(contingency_table_pd)

    total_purity = sum(contingency_table.max(axis=1)) / df_filtered.shape[0]

    ari_score = adjusted_rand_score(df_filtered["label"], df_filtered["cluster"])

    return dict(NMI=nmi_score, purity=total_purity, ari=ari_score)
