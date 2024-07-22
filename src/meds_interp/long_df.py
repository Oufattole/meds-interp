import polars as pl
import numpy as np
from datetime import datetime


def generate_long_df(df: pl.DataFrame, id_column="patient_id", timestamp_column="timestamp") -> pl.DataFrame:
    """Makes a long form version of the given dataframe input df has columns "patient_id", "timestamp",
    "label", "embedding_1" all the way to "embedding_n"."""
    long_rows = []
    embeddings = df.columns[5:]
    for row in df.iter_rows(named=True):
        for embedding in embeddings:
            i = 0
            if row[embedding] is None:
                continue
            elif isinstance(row[embedding], (int, float, str)):
                long_rows.append(
                    {
                        id_column: row[id_column],
                        timestamp_column: row[timestamp_column],
                        "code": f"{embedding}",
                        "numerical_value": row[embedding],
                    }
                )
            elif isinstance(row[embedding], datetime):
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

def generate_long_df_explode(df: pl.DataFrame, id_column="patient_id", timestamp_column="timestamp", \
                             label_column = "label") -> pl.DataFrame:
    """Makes a long form version of the given dataframe input df has columns "patient_id", "timestamp",
    "label", "embedding_1" all the way to "embedding_n" using the polars explode function."""

    embeddings = list(df.collect_schema().names())[5:]
    embedding_types = list(df.collect_schema().dtypes())[5:]

    
    # Loop through the columns and explode the columns that are lists
    # but only keep one clumn at a time and add it to a vstack of all the columns

    long_df = pl.LazyFrame()
    import pdb; pdb.set_trace()
    height = df.select(pl.len()).collect().item()
    df.shape[0]
    
    for embedding, embedding_type in zip(embeddings, embedding_types):
        exploded = df.select(list(df.collect_schema().names())[:5] + [embedding])
        exploded = exploded.drop_nulls()
        if isinstance(embedding_type, pl.Array):
            import pdb; pdb.set_trace()
            length = list(exploded.select(embedding).collect_schema().dtypes())[0].shape[0]
            exploded = exploded.with_columns(
                pl.Series([np.arange(length)] * height).alias("index")
            )
            import pdb; pdb.set_trace()
            exploded = exploded.explode([embedding, "index"])
            # exploded = exploded.with_columns(
            #     exploded.with_columns(pl.lit(embedding).alias("code")),
            # )
            exploded = exploded.with_columns(
                pl.lit(embedding).alias("code"),
            )
            exploded = exploded.rename({embedding: "numerical_value"})
            exploded = exploded.select([id_column, timestamp_column, "code", "index", "numerical_value"])
            # exploded = exploded.drop_nulls()
            # exploded = exploded.with_columns(pl.col("numerical_value").cast(pl.Float32))
            long_df = pl.concat([long_df, exploded], how="vertical")
            #change to concat and vertical
        else:
            # exploded = exploded.with_columns(pl.col(embedding).cast(pl.Float32))
            # exploded = exploded.with_columns(
            #     exploded.with_columns(pl.lit(embedding).alias("code")),
            # )
            exploded = exploded.with_columns(
                pl.lit(embedding).alias("code"),
                pl.Series(np.zeros(height)).alias("index")
            )
            exploded = exploded.rename({embedding: "numerical_value"})
            exploded = exploded.select([id_column, timestamp_column, "code", "index", "numerical_value"])
            exploded = exploded.drop_nulls()
            long_df = pl.concat([long_df, exploded], how="vertical")

    return long_df



if __name__ == "__main__":
    data_1 = {
        "patient_id": [1, 1, 1, 2, 2, 2, 3],
        "timestamp": [1, 2, 3, 1, 2, 3, 1],
        "label": [0, 1, 1, 0, 1, 1, 1],
        "business_id": [1, 1, 1, 2, 2, 2, 3],
        "user_id": [1, 1, 1, 2, 2, 2, 3],
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
    df = pl.LazyFrame(data_1)
    # print(df)
    long_df = generate_long_df_explode(df)
    print(long_df)
