import polars as pl


def generate_long_df(df: pl.DataFrame) -> pl.DataFrame:
    """Makes a long form version of the given dataframe input df has columns "patient ID", "time", "label",
    "embedding_1" all the way to "embedding_n"."""
    long_rows = []
    embeddings = df.columns[3:]
    for row in df.iter_rows(named=True):
        for embedding in embeddings:
            i = 0
            if row[embedding] is None:
                continue
            for value in row[embedding]:
                long_rows.append(
                    {
                        "patient_id": row["patient_id"],
                        "timestamp": row["timestamp"],
                        "code": f"{embedding}_{i}",
                        "numerical_value": value,
                    }
                )
                i += 1

    long_df = pl.DataFrame(long_rows)
    return long_df


if __name__ == "__main__":
    data_1 = {
        "Patient_ID": [1, 1, 1, 2, 2, 2, 3],
        "Time": [1, 2, 3, 1, 2, 3, 1],
        "Label": [0, 1, 1, 0, 1, 1, 1],
        "Embedding_1": [
            [0.1, 0.2, 0.2],
            [0.3, 0.5, 0.2],
            [0.5, 0.5, 0.1],
            [0.0, 0.4, 0.6],
            [0.7, 0.1, 0.2],
            [0.2, 0.2, 0.5],
            [0.7, 0.4, 0.1],
        ],
        "Embedding_2": [[0.5, 0.6], [0.7, 0.8], [0.1, 0.4], [0.3, 0.2], [0.4, 0.3], [0.1, 0.3], [0.4, 0.1]],
    }
    data_2 = {
        "Patient_ID": [1, 1, 2],
        "Time": [1, 2, 1],
        "Label": [0, 1, 1],
        "Embedding_1": [[0.1, 0.2], [0.3, 0.4], [0.2, 0.4]],
        "Embedding_2": [[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]],
    }
    df = pl.DataFrame(data_2)
    print(df)
    long_df = generate_long_df(df)
    print(long_df)
