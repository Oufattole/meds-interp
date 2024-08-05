from pathlib import Path

import polars as pl
from hydra import compose, initialize

from meds_interp import knn_raw_code as knn
from meds_interp import meds_logistic_regression as lr


def test_lr():
    train_df = pl.DataFrame({"embeddings": [[1, 1, 0], [1, 2, 1], [2, 0, -2]], "label": [0, 1, 0]})
    test_df = pl.DataFrame({"embeddings": [[1, 2, 1], [1, 0, 0], [1, 0, 1]], "label": [1, 0, 1]})
    probabilities = lr.fit_logistic_regression(train_df, test_df, 1.0)
    test_df = test_df.with_columns(pl.Series(probabilities).alias("probabilities"))
    lr.score_labels(test_df)


def test_knn_model():
    train_df_2 = pl.DataFrame(
        {
            "modality_1": [[1, 1, 0], [1, 2, 1], [2, 0, -2], [0, 1, 2]],
            "modality_2": [[1, 0, 0], [0, 2, 1], [4, 1, 1], [3, 1, 2]],
            "label": [0, 1, 0, 0],
        }
    )

    modalities = ["modality_1", "modality_2"]
    modality_weights = [4, 1]

    test1 = knn.KNN_Model(modalities=modalities, modality_weights=modality_weights, n_neighbors=2)
    test1.fit(train_df_2)
    x1 = test1.predict(train_df_2)
    assert x1.shape == train_df_2.get_column("label").shape

    preprocess2 = knn.Preprocess_Type.NORM_AFTER_CONCAT
    test2 = knn.KNN_Model(modalities=modalities, modality_weights=modality_weights, preprocess=preprocess2)
    test2.fit(train_df_2)
    x2 = test2.predict(train_df_2)
    assert x2.shape == train_df_2.get_column("label").shape

    preprocess3 = knn.Preprocess_Type.NORM_SEPERATLY
    test3 = knn.KNN_Model(modalities=modalities, modality_weights=modality_weights, preprocess=preprocess3)
    test3.fit(train_df_2)
    x3 = test3.predict(train_df_2)
    assert x3.shape == train_df_2.get_column("label").shape


# def test_knn_tuning(tmp_path):
#     train_df = pl.DataFrame(
#         {
#             "modality_1": [[1, 1, 0], [1, 2, 1], [2, 0, -2], [0, 1, 2]],
#             "modality_2": [[1, 0, 0], [0, 2, 1], [4, 1, 1], [3, 1, 2]],
#             "modality_3": [1, 2, 3, 4],
#             "label": [0, 0, 1, 1],
#         }
#     )
#     val_df = pl.DataFrame(
#         {
#             "modality_1": [[3, 5, 2], [4, 1, 3], [6, 2, 0], [7, 1, 1]],
#             "modality_2": [[2, 3, 1], [5, 0, 4], [1, 6, 2], [3, 2, 5]],
#             "modality_3": [2, 6, 1, 1],
#             "label": [0, 1, 0, 1],
#         }
#     )
#     test_df = pl.DataFrame(
#         {
#             "modality_1": [[-2, 3, -1], [0, -1, 2], [-3, 1, -2], [1, -2, 3]],
#             "modality_2": [[3, -1, 0], [-2, 2, -3], [1, -3, 2], [0, 3, -1]],
#             "modality_3": [1, 4, 2, 6],
#             "label": [0, 0, 1, 1],
#         }
#     )
#     train_df.write_parquet(Path(tmp_path) / "train.parquet")
#     val_df.write_parquet(Path(tmp_path) / "val.parquet")
#     test_df.write_parquet(Path(tmp_path) / "test.parquet")

#     # import pdb; pdb.set_trace()

#     test_config = {
#         "modalities": ["modality_1", "modality_2", "modality_3"],
#         "+weights.modality_1": 1,
#         "+weights.modality_2": 1,
#         "+weights.modality_3": 1,
#         "input_path": tmp_path,
#     }

#     with initialize(version_base=None, config_path="../src/meds_interp/configs/"):  # path to config.yaml
#         overrides = [f"{k}={v}" for k, v in test_config.items()]
#         cfg = compose(config_name="knn", overrides=overrides)  # config.yaml
#     knn.main(cfg)

def test_yelp_knn(tmp_path):
    total_rows = 200000
    yelp_lazy = pl.scan_parquet("/home/shared/yelp/merged_df.parquet").rename({"stars": "label"}).head(total_rows)

    modalities = ['review_embeddings', 'useful', 'funny', 'cool', 'user_review_count', 'useful_sent', 'funny_sent', 'cool_sent', \
     'fans', 'num_years_elite', 'average_stars_given', 'compliment_hot', 'compliment_more', 'compliment_profile', \
    'compliment_cute', 'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool', \
    'compliment_funny', 'compliment_writer', 'compliment_photos', 'attribute_embeddings', 'category_embeddings', \
    'avg_stars', 'bus_review_count', 'cap_embeddings_outside', 'cap_embeddings_drink', 'cap_embeddings_food', \
    'cap_embeddings_menu', 'cap_embeddings_inside', 'img_embeddings_drink', 'img_embeddings_food', 'img_embeddings_menu', \
    'img_embeddings_outside', 'img_embeddings_inside']
    empty_vector_fix = [
        pl.when(pl.col(modality).is_null() | (pl.col(modality) == []))
            .then([0.0]*512)
            .otherwise(pl.col(modality))
            .alias(modality)
        for modality in ['attribute_embeddings', 'category_embeddings', 'cap_embeddings_outside', 'cap_embeddings_drink', 'cap_embeddings_food',
                         'cap_embeddings_menu', 'cap_embeddings_inside', 'img_embeddings_drink', 'img_embeddings_food',
                         'img_embeddings_menu', 'img_embeddings_outside', 'img_embeddings_inside']
    ]
    empty_int_fix = [
        pl.when(pl.col(modality).is_null())
            .then(0)
            .otherwise(pl.col(modality))
            .alias(modality)
        for modality in ['avg_stars', 'bus_review_count']
        # , 'fans', 'num_years_elite', 'average_stars_given', 'compliment_hot',
        #                  'compliment_more', 'compliment_profile', 'compliment_cute', 'compliment_list', 'compliment_note',
        #                  'compliment_plain', 'compliment_cool', 'compliment_funny', 'compliment_writer', 'compliment_photos']
    ]
    yelp_lazy = yelp_lazy.with_columns(empty_int_fix)
    yelp_lazy = yelp_lazy.with_columns(empty_vector_fix)
    yelp_lazy = yelp_lazy.with_columns((pl.when(pl.col("label") >= 3)
                                          .then(1)
                                          .otherwise(0)
                                          .alias("label")))
    # yelp_lazy = yelp_lazy.with_columns((pl.col("label") - 1).alias("label"))
    
    # train_df = yelp_lazy.slice(0, 4194168)
    # val_df = yelp_lazy.slice(4194168, 1398056)
    # test_df = yelp_lazy.slice(5592224, 1398056)

    height = yelp_lazy.select(pl.len()).collect().item()
    # total_rows = height

    # Calculate the number of rows for each split
    train_rows = int(total_rows * 0.6)
    val_rows = int(total_rows * 0.2)
    test_rows = int(total_rows * 0.2)

    # Create slices for each split
    train_df = yelp_lazy.slice(0, train_rows)
    val_df = yelp_lazy.slice(train_rows, val_rows)
    test_df = yelp_lazy.slice(train_rows + val_rows, test_rows)

    train_df.collect().write_parquet(Path(tmp_path) / "train.parquet")
    val_df.collect().write_parquet(Path(tmp_path) / "val.parquet")
    test_df.collect().write_parquet(Path(tmp_path) / "test.parquet")

    # import pdb; pdb.set_trace()
    test_config = {
        "modalities": modalities,
        "input_path": tmp_path,
    }

    for modality in modalities:
        test_config[f"+weights.{modality}"] = 1
    
    with initialize(version_base=None, config_path="../src/meds_interp/configs/"):  # path to config.yaml
        overrides = [f"{k}={v}" for k, v in test_config.items()]
        cfg = compose(config_name="knn", overrides=overrides)  # config.yaml
    knn.main(cfg)