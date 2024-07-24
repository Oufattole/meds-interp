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
                             label_column = "label", num_idx_cols = 5, more_ids = False) -> pl.DataFrame:
    """Makes a long form version of the given dataframe input df has columns "patient_id", "timestamp",
    "label", "embedding_1" all the way to "embedding_n" using the polars explode function."""

    embeddings = list(df.collect_schema().names())[num_idx_cols:]
    embedding_types = list(df.collect_schema().dtypes())[num_idx_cols:]

    if timestamp_column == None:
        timestamp_bool = False
    else:
        timestamp_bool = True
    first_pass = True
    # Loop through the columns and explode the columns that are lists
    # but only keep one clumn at a time and add it to a vstack of all the columns

    # long_df = pl.LazyFrame()
    # import pdb; pdb.set_trace()
    height = df.select(pl.len()).collect().item()
    # df.shape[0]
    if more_ids:
        ids_list = []
        for col in df.collect_schema().names():
            if "id" in col:
                ids_list.append(col)
    # import pdb
    for embedding, embedding_type in zip(embeddings, embedding_types):
        exploded = df.select(list(df.collect_schema().names())[:num_idx_cols] + [embedding])
        exploded = exploded.drop_nulls()
        # pdb.set_trace()
        if isinstance(embedding_type, pl.Array):
            length = list(exploded.select(embedding).collect_schema().dtypes())[0].shape[0]
            index_array = np.array([np.arange(length)] * height)
            exploded = exploded.collect()
            exploded = exploded.with_columns(
                pl.Series(index_array).alias("index"),
                pl.Series([embedding] * (height)).alias("code"),
            )
            # pdb.set_trace()
            exploded = exploded.lazy()
            exploded = exploded.explode([embedding, "index"])
            # pdb.set_trace()
            # exploded = exploded.with_columns(exploded.with_columns(pl.lit(embedding).alias("code")))
            # exploded = exploded.with_columns(
            #     # pl.lit(embedding).alias("code"),
            #     pl.Series([embedding] * (height * length)).alias("code"),
            # )
            exploded = exploded.rename({embedding: "numerical_value"})
            # pdb.set_trace()
            if timestamp_bool:
                if more_ids:
                    exploded = exploded.select(ids_list + [timestamp_column, "code", "index", "numerical_value"])
                else:
                    exploded = exploded.select([id_column, timestamp_column, "code", "index", "numerical_value"])
            else:
                if more_ids:
                    exploded = exploded.select(ids_list + ["code", "index", "numerical_value"])
                else:
                    exploded = exploded.select([id_column, "code", "index", "numerical_value"])
            # exploded = exploded.drop_nulls()
            exploded = exploded.with_columns(
                pl.col("numerical_value").cast(pl.Float32),
                pl.col("index").cast(pl.Int32)
                )
            # long_df = pl.concat([long_df, exploded], how="vertical")
            #change to concat and vertical
        else:
            # exploded = exploded.with_columns(pl.col(embedding).cast(pl.Float32))
            # exploded = exploded.with_columns(
            #     exploded.with_columns(pl.lit(embedding).alias("code")),
            # )
            exploded = exploded.collect()
            exploded = exploded.with_columns(
                # pl.lit(embedding).alias("code"),
                pl.Series([embedding] * height).alias("code"),
                pl.Series(np.zeros(height, dtype=np.int64)).alias("index")
            )
            exploded = exploded.lazy()
            exploded = exploded.rename({embedding: "numerical_value"})
            # exploded = exploded.select([id_column, timestamp_column, "code", "index", "numerical_value"])
            # exploded = exploded.drop_nulls()
            if timestamp_bool:
                if more_ids:
                    exploded = exploded.select(ids_list + [timestamp_column, "code", "index", "numerical_value"])
                else:
                    exploded = exploded.select([id_column, timestamp_column, "code", "index", "numerical_value"])
            else:
                if more_ids:
                    exploded = exploded.select(ids_list + ["code", "index", "numerical_value"])
                else:
                    exploded = exploded.select([id_column, "code", "index", "numerical_value"])
            exploded = exploded.with_columns(
                pl.col("numerical_value").cast(pl.Float32),
                pl.col("index").cast(pl.Int32)
                )
        # import pdb; pdb.set_trace()
        if first_pass:
            empty_data = {col_name: pl.Series([], dtype=col_type) for col_name, col_type in exploded.collect_schema().items()}
            long_df = pl.LazyFrame(empty_data)
            first_pass = False
        long_df = pl.concat([long_df, exploded], how="vertical")
        # import pdb; pdb.set_trace()

    return long_df



if __name__ == "__main__":
    data_1 = {
        "patient_id": np.array([1, 1, 1, 2, 2, 2, 3]),
        # "timestamp": np.array([1, 2, 3, 1, 2, 3, 1]),
        "label": np.array([0, 1, 1, 0, 1, 1, 1]),
        "business_id": np.array([1, 1, 1, 2, 2, 2, 3]),
        "user_id": np.array([1, 1, 1, 2, 2, 2, 3]),
        "Embedding_1": np.array([
            [0.1, 0.2, 0.2],
            [0.3, 0.5, 0.2],
            [0.5, 0.5, 0.1],
            [0.0, 0.4, 0.6],
            [0.7, 0.1, 0.2],
            [0.2, 0.2, 0.5],
            [0.7, 0.4, 0.1],
        ]),
        "Embedding_2": np.array([[0.5, 0.6], [0.7, 0.8], [0.1, 0.4], [0.3, 0.2], [0.4, 0.3], [0.1, 0.3], [0.4, 0.1]]),
        # "Embedding_3": np.array([1, 2, 3, 4, 5, 6, 7]),
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
    print(df.collect())
    long_df = generate_long_df_explode(df, timestamp_column=None, num_idx_cols=4)
    long_df.sink_parquet("/home/leander/projects/meds-interp/src/meds_interp/test.parquet")
    print(long_df.collect())
