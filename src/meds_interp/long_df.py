import polars as pl


def generate_long_df(df: pl.DataFrame, id_column="patient_id", timestamp_column="timestamp") -> pl.DataFrame:
    """Makes a long form version of the given dataframe input df has columns "patient_id", "timestamp",
    "label", "embedding_1" all the way to "embedding_n"."""
    long_rows = []
    embeddings = df.columns[3:]
    for row in df.iter_rows(named=True):
        for embedding in embeddings:
            i = 0
            if row[embedding] is None:
                continue
            elif isinstance(row[embedding], int):
                long_rows.append(
                    {
                        id_column: row[id_column],
                        timestamp_column: row[timestamp_column],
                        "code": f"{embedding}",
                        "numerical_value": row[embedding],
                    }
                )
            else:
                for value in row[embedding]:
                    long_rows.append(
                        {
                            id_column: row[id_column],
                            timestamp_column: row[timestamp_column],
                            "code": f"{embedding}_{i}",
                            "numerical_value": value,
                        }
                    )
                    i += 1

    long_df = pl.DataFrame(long_rows)
    return long_df


if __name__ == "__main__":
    data_1 = {
        "patient_id": [1, 1, 1, 2, 2, 2, 3],
        "timestamp": [1, 2, 3, 1, 2, 3, 1],
        "label": [0, 1, 1, 0, 1, 1, 1],
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
        "patient_id": [1, 1, 2],
        "timestamp": [1, 2, 1],
        "label": [0, 1, 1],
        "Embedding_1": [[0.1, 0.2], [0.3, 0.4], [0.2, 0.4]],
        "Embedding_2": [[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]],
        "Embedding_3": [1, 2, 3],
    }
    df = pl.DataFrame(data_2)
    print(df)
    long_df = generate_long_df(df)
    print(long_df)
