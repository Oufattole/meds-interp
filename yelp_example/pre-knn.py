import hashlib
import json
import os
from pathlib import Path

import numpy as np
import polars as pl
import torch
from loguru import logger
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

from meds_interp.long_df import generate_long_df_explode

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = processor.tokenizer
image_processor = processor.image_processor

device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda"
model.to(device)

filepaths = [
    "/mnt/hdd/shared/shared/yelp_dataset/raw/yelp_academic_dataset_business.json",
    "/mnt/hdd/shared/shared/yelp_dataset/raw/yelp_academic_dataset_user.json",
    "/mnt/hdd/shared/shared/yelp_dataset/raw/yelp_academic_dataset_review.json",
    "/mnt/hdd/shared/shared/yelp_dataset_image/photos.json",
]
photo_table_output_path = "/home/shared/yelp/photo_table.parquet"
caption_table_output_path = "/home/shared/yelp/caption_table.parquet"


def hash_id(uid):
    byte_string = uid.encode()
    return int.from_bytes(hashlib.sha256(byte_string).digest()[:4], "little")


def process_photo_captions():
    photo_caption_labels = ["drink", "food", "inside", "menu", "outside"]
    statement = True
    for label in photo_caption_labels:
        statement and os.path.exists(f"/home/shared/yelp/caption_table_{label}.parquet") and os.path.exists(
            f"/home/shared/yelp/photo_table_{label}.parquet"
        )
    # if not statement:
    with open("/home/shared/yelp/raw/yelp_dataset_image/photos.json") as r:
        photo_ids = []
        photo_business_ids = []
        photo_labels = []
        captions = []
        caption_labels = []
        caption_business_ids = []
        for line in r:
            data = json.loads(line)
            try:
                img_path = "/home/shared/yelp/raw/yelp_dataset_image" + "/photos/" + data["photo_id"] + ".jpg"
                Image.open(img_path)
                photo_ids.append(data["photo_id"])
                photo_business_ids.append(data["business_id"])
                photo_labels.append(data["label"])
                if data["caption"]:
                    captions.append(data["caption"])
                    caption_labels.append(data["label"])
                    caption_business_ids.append(data["business_id"])
            except UnidentifiedImageError:
                continue

    class PhotoDataset(Dataset):
        def __init__(self, photo_ids_list: list[str]):
            """Dataset of Photos.

            Args:
                imgs_paths_list (list[str]): paths to each photo we want an embedding for
            """
            self.photo_ids_list = photo_ids_list

        def __len__(self):
            return len(self.photo_ids_list)

        def __getitem__(self, idx):
            img_path = (
                "/home/shared/yelp/raw/yelp_dataset_image" + "/photos/" + self.photo_ids_list[idx] + ".jpg"
            )
            image = Image.open(img_path)
            image_tensor = image_processor(image, return_tensors="pt")
            return image_tensor

    # photo_output_path = "/home/shared/yelp/photos.npy"
    photo_output_path = "/home/leander/projects/meds-interp/photos.npy"
    if not os.path.exists(photo_output_path):
        photo_ids_list = photo_ids
        photo_dataloader = DataLoader(
            PhotoDataset(photo_ids_list), shuffle=False, batch_size=64, num_workers=8
        )
        photo_all_embeddings = []
        for batch in tqdm(photo_dataloader):
            batch.to(device)
            with torch.no_grad():
                embeddings = model.get_image_features(pixel_values=batch["pixel_values"].squeeze(dim=1))
                photo_all_embeddings.append(embeddings.detach().cpu().numpy())
        photo_stacked_embeddings = np.vstack(photo_all_embeddings)
        np.save(photo_output_path, photo_stacked_embeddings)

    # Dataframe of the image embeddings and the labels
    photo_table = {
        "business_id": [hash(bus_id) for bus_id in photo_business_ids],
        "image_embeddings": np.load(photo_output_path),
        "labels": photo_labels,
    }
    photo_table = pl.DataFrame(photo_table)

    unique_photo_labels = photo_table["labels"].unique().to_list()
    for label in unique_photo_labels:
        new_table = photo_table.filter(pl.col("labels") == label)
        new_table = new_table.rename({"image_embeddings": f"image_embeddings_{label}"})
        new_table = new_table.drop(["labels"])

        new_table.write_parquet(f"/home/shared/yelp/photo_table_{label}.parquet")
        # photo_table = photo_table.with_columns(
        #     pl.when(pl.col("labels") == label)
        #     .then(pl.col("image_embeddings"))
        #     .otherwise([])
        #     .alias(f"img_embeddings_{label}")
        # )
    # pdb.set_trace()
    # photo_table = photo_table.drop(["image_embeddings", "labels"])
    # photo_table.write_parquet(photo_table_output_path)

    class CaptionDataset(Dataset):
        def __init__(self, photo_captions_list: list[str]):
            """Dataset of the captions.

            Args:
                photo_captions_list (list[str]): list of all the captions
            """
            self.photo_captions_list = photo_captions_list

        def __len__(self):
            return len(self.photo_captions_list)

        def __getitem__(self, idx):
            return self.photo_captions_list[idx]

    # caption_output_path = "/home/shared/yelp/captions.npy"
    caption_output_path = "/home/leander/projects/meds-interp/captions.npy"
    if not os.path.exists(caption_output_path):
        captions_dataloader = DataLoader(CaptionDataset(captions), shuffle=False, batch_size=64)
        caption_all_embeddings = []
        for batch in tqdm(captions_dataloader):
            tokens = tokenizer(
                batch, max_length=77, padding="max_length", truncation=True, return_tensors="pt"
            )
            tokens = {key: value.to(model.device) for key, value in tokens.items()}

            with torch.no_grad():
                embeddings = model.get_text_features(**tokens)
                caption_all_embeddings.append(embeddings.cpu().numpy())
        caption_stacked_embeddings = np.vstack(caption_all_embeddings)
        np.save(caption_output_path, caption_stacked_embeddings)

    # Dataframe of the caption embeddings and the labels
    photo_caption_table = {
        "business_id": [hash(bus_id) for bus_id in caption_business_ids],
        "captions": np.load(caption_output_path),
        "labels": caption_labels,
    }
    photo_caption_table = pl.DataFrame(photo_caption_table)
    unique_caption_labels = photo_caption_table["labels"].unique().to_list()
    for label in unique_caption_labels:
        new_table = photo_caption_table.filter(pl.col("labels") == label)
        new_table = new_table.rename({"captions": f"captions_{label}"})
        new_table = new_table.drop(["labels"])

        new_table.write_parquet(f"/home/shared/yelp/caption_table_{label}.parquet")
    #     photo_caption_table = photo_caption_table.with_columns(
    #         pl.when(pl.col("labels") == label)
    #         .then(pl.col("captions"))
    #         .otherwise(np.array([]))
    #         .alias(f"cap_embeddings_{label}")
    #     )
    # photo_caption_table = photo_caption_table.drop(["captions", "labels"])
    # photo_caption_table.write_parquet(caption_table_output_path)


# Load all of the reviews into a dataframe
reviews_table_output_path = "/home/shared/yelp/reviews_table.parquet"


def process_review():
    # if (
    #     not os.path.exists(reviews_table_output_path)
    #     or pl.scan_parquet(reviews_table_output_path, n_rows=1).collect().is_empty()
    # ):
    with open("/home/shared/yelp/raw/yelp_academic_dataset_review.json") as r:
        review_id = []
        review_user = []
        review = []
        review_business = []
        timestamps = []
        useful_count = []
        funny_count = []
        cool_count = []
        stars = []
        for line in r:
            data = json.loads(line)

            review_id.append(data["review_id"])
            review.append(data["text"])
            review_user.append(data["user_id"])
            review_business.append(data["business_id"])
            timestamps.append(data["date"])
            useful_count.append(data["useful"])
            funny_count.append(data["funny"])
            cool_count.append(data["cool"])
            stars.append(int(data["stars"]))

    class ReviewDataset(Dataset):
        def __init__(self, reviews_list: list[str]):
            """Dataset of the reviews.

            Args:
                reviews_list (list[str]): list of all the reviews
            """
            self.reviews_list = reviews_list

        def __len__(self):
            return len(self.reviews_list)

        def __getitem__(self, idx):
            return self.reviews_list[idx]

    # review_output_path = "/home/shared/yelp/reviews.npy"
    review_output_path = "/home/leander/projects/meds-interp/reviews.npy"
    model.eval()
    if not os.path.exists(review_output_path):
        review_dataloader = DataLoader(ReviewDataset(review), shuffle=False, batch_size=1024)
        review_all_embeddings = []
        for batch in tqdm(review_dataloader):
            tokens = tokenizer(
                batch, max_length=77, padding="max_length", truncation=True, return_tensors="pt"
            )
            tokens = {key: value.to(model.device) for key, value in tokens.items()}

            with torch.no_grad():
                embeddings = model.get_text_features(**tokens)
                review_all_embeddings.append(embeddings.cpu().numpy())
        review_stacked_embeddings = np.vstack(review_all_embeddings)
        np.save(review_output_path, review_stacked_embeddings)

    reviews_table = {
        # "review_id": [hash_id(rev_id) for rev_id in review_id],
        # "review_id": review_id,
        "review_id": [hash(rev_id) for rev_id in review_id],
        "user_id": [hash(use_id) for use_id in review_user],
        "timestamp": np.array(timestamps, dtype="datetime64[ns]"),
        "business_id": [hash(bus_id) for bus_id in review_business],
        "stars": stars,
        "review_embeddings": np.load(review_output_path),
        "useful": useful_count,
        "funny": funny_count,
        "cool": cool_count,
    }
    reviews_table = pl.DataFrame(reviews_table)
    reviews_table.write_parquet(reviews_table_output_path)


business_table_output_path = "/home/shared/yelp/business_table.parquet"


def process_business():
    # if (
    #     not os.path.exists(business_table_output_path)
    #     or pl.scan_parquet(business_table_output_path, n_rows=1).collect().is_empty()
    # ):
    with open("/home/shared/yelp/raw/yelp_academic_dataset_business.json") as r:
        business_id = []
        avg_stars = []
        review_count = []
        attributes = []
        categories = []
        for line in r:
            data = json.loads(line)
            if isinstance(data["categories"], str):
                business_id.append(data["business_id"])
                avg_stars.append(data["stars"])
                review_count.append(data["review_count"])
                attributes.append(str(data["attributes"]))
                categories.append(data["categories"])
            else:
                continue

    class BusinessDataset(Dataset):
        def __init__(self, business_list: list[str]):
            """Dataset of the businesses.

            Args:
                reviews_list (list[str]): list of all the business categories and attributes
            """
            self.business_list = business_list

        def __len__(self):
            return len(self.business_list)

        def __getitem__(self, idx):
            return self.business_list[idx]

    # business_categories_output_path = "/home/shared/yelp/business_categories.npy"
    business_categories_output_path = "/home/leander/projects/meds-interp/business_categories.npy"
    model.eval()
    if not os.path.exists(business_categories_output_path):
        business_cat_dataloader = DataLoader(BusinessDataset(categories), shuffle=False, batch_size=64)
        business_cat_all_embeddings = []
        for batch in tqdm(business_cat_dataloader):
            tokens = tokenizer(
                batch, max_length=77, padding="max_length", truncation=True, return_tensors="pt"
            )
            tokens = {key: value.to(model.device) for key, value in tokens.items()}

            with torch.no_grad():
                embeddings = model.get_text_features(**tokens)
                business_cat_all_embeddings.append(embeddings.cpu().numpy())
        business_cat_stacked_embeddings = np.vstack(business_cat_all_embeddings)
        np.save(business_categories_output_path, business_cat_stacked_embeddings)

    # business_attributes_output_path = "/home/shared/yelp/business_attributes.npy"
    business_attributes_output_path = "/home/leander/projects/meds-interp/business_attributes.npy"
    model.eval()
    if not os.path.exists(business_attributes_output_path):
        business_att_dataloader = DataLoader(BusinessDataset(attributes), shuffle=False, batch_size=64)
        business_att_all_embeddings = []
        for batch in tqdm(business_att_dataloader):
            tokens = tokenizer(
                batch, max_length=77, padding="max_length", truncation=True, return_tensors="pt"
            )
            tokens = {key: value.to(model.device) for key, value in tokens.items()}

            with torch.no_grad():
                embeddings = model.get_text_features(**tokens)
                business_att_all_embeddings.append(embeddings.cpu().numpy())
        business_att_stacked_embeddings = np.vstack(business_att_all_embeddings)
        np.save(business_attributes_output_path, business_att_stacked_embeddings)

    business_table = {
        "business_id": [hash(bus_id) for bus_id in business_id],
        "attribute_embeddings": np.load(business_attributes_output_path),
        "category_embeddings": np.load(business_categories_output_path),
        "avg_stars": avg_stars,
        "bus_review_count": review_count,
    }
    business_table = pl.DataFrame(business_table)
    business_table.write_parquet(business_table_output_path)


user_table_output_path = "/home/shared/yelp/user_table.parquet"


def process_user():
    # if (
    #     not os.path.exists(user_table_output_path)
    #     or pl.scan_parquet(user_table_output_path, n_rows=1).collect().is_empty()
    # ):
    with open("/home/shared/yelp/raw/yelp_academic_dataset_user.json") as r:
        user_id = []
        yelping_since = []
        review_count = []
        useful_sent = []
        funny_sent = []
        cool_sent = []
        fans = []
        elite = []
        average_stars = []
        compliment_hot = []
        compliment_more = []
        compliment_profile = []
        compliment_cute = []
        compliment_list = []
        compliment_note = []
        compliment_plain = []
        compliment_cool = []
        compliment_funny = []
        compliment_writer = []
        compliment_photos = []

        for line in r:
            data = json.loads(line)

            user_id.append(data["user_id"])
            yelping_since.append(data["yelping_since"])
            review_count.append(data["review_count"])
            useful_sent.append(data["useful"])
            funny_sent.append(data["funny"])
            cool_sent.append(data["cool"])
            fans.append(data["fans"])
            elite.append(len(data["elite"]))
            average_stars.append(data["average_stars"])
            compliment_hot.append(data["compliment_hot"])
            compliment_more.append(data["compliment_more"])
            compliment_profile.append(data["compliment_profile"])
            compliment_cute.append(data["compliment_cute"])
            compliment_list.append(data["compliment_list"])
            compliment_note.append(data["compliment_note"])
            compliment_plain.append(data["compliment_plain"])
            compliment_cool.append(data["compliment_cool"])
            compliment_funny.append(data["compliment_funny"])
            compliment_writer.append(data["compliment_writer"])
            compliment_photos.append(data["compliment_photos"])

    user_table = {
        "user_id": [hash(use_id) for use_id in user_id],
        "yelping_since": np.array(yelping_since, dtype="datetime64[ns]"),
        "user_review_count": review_count,
        "useful_sent": useful_sent,
        "funny_sent": funny_sent,
        "cool_sent": cool_sent,
        "fans": fans,
        "num_years_elite": elite,
        "average_stars_given": average_stars,
        "compliment_hot": compliment_hot,
        "compliment_more": compliment_more,
        "compliment_profile": compliment_profile,
        "compliment_cute": compliment_cute,
        "compliment_list": compliment_list,
        "compliment_note": compliment_note,
        "compliment_plain": compliment_plain,
        "compliment_cool": compliment_cool,
        "compliment_funny": compliment_funny,
        "compliment_writer": compliment_writer,
        "compliment_photos": compliment_photos,
    }
    user_table = pl.DataFrame(user_table)
    user_table.write_parquet(user_table_output_path)


def aggregate_df():
    def mean_vector(series):
        vectors = [vector for vector in series.to_list() if vector is not None]
        if vectors:
            mean_vec = np.mean(vectors, axis=0).tolist()
            return mean_vec
        else:
            return []

    # review_user_grouped = process_review().group_by("user_id", maintain_order=True).agg([
    #     pl.col("review_embeddings").map_elements(mean_vector, return_dtype=pl.List(pl.Float64)),
    #     pl.col("useful").mean(),
    #     pl.col("funny").mean(),
    #     pl.col("cool").mean(),
    #     pl.col("stars").mean(),
    # ])
    # user_review_joined = review_user_grouped.join(process_user(), on="user_id", how="left", coalesce=True)
    # print(user_review_joined)
    # review_business_grouped = process_review().group_by("business_id", maintain_order=True).agg([
    #     pl.col("review_embeddings").map_elements(mean_vector, return_dtype=pl.List(pl.Float64)),
    #     pl.col("useful").mean(),
    #     pl.col("funny").mean(),
    #     pl.col("cool").mean(),
    #     pl.col("stars").mean(),
    # ])
    # # print(review_business_grouped)

    # photos_grouped = (
    #     process_photos()
    #     .group_by("business_id", maintain_order=True)
    #     .agg(
    #         [
    #             pl.col("embeddings_inside").map_elements(mean_vector, return_dtype=pl.List(pl.Float64)),
    #             pl.col("embeddings_outside").map_elements(mean_vector, return_dtype=pl.List(pl.Float64)),
    #             pl.col("embeddings_food").map_elements(mean_vector, return_dtype=pl.List(pl.Float64)),
    #             pl.col("embeddings_drink").map_elements(mean_vector, return_dtype=pl.List(pl.Float64)),
    #             pl.col("embeddings_menu").map_elements(mean_vector, return_dtype=pl.List(pl.Float64)),
    #         ]
    #     )
    # )
    # for label in ["inside", "outside", "food", "drink", "menu"]:
    #     photos_grouped = photos_grouped.with_columns(
    #         pl.when(pl.col(f"embeddings_{label}") == [])
    #         .then(None)
    #         .otherwise(pl.col(f"embeddings_{label}"))
    #         .alias(f"embeddings_{label}")
    #     )
    # # print(photos_grouped)
    # print(photo_review_joined)
    logger.info("Reading Reviews")
    review_df = pl.scan_parquet(reviews_table_output_path)
    # import pdb; pdb.set_trace()
    # review_df = review_df.head(100)
    logger.info("Reading Users")
    user_df = pl.scan_parquet(user_table_output_path).drop(["yelping_since"])
    logger.info("Reading Captions")
    photo_caption_labels = ["drink", "food", "inside", "menu", "outside"]
    caption_tables = []
    for label in photo_caption_labels:
        caption_tables.append(pl.scan_parquet(f"/home/shared/yelp/caption_table_{label}.parquet"))
    logger.info(f"Reviews_df.shape: {review_df.select(pl.len()).collect().item()}")
    logger.info("Reading Photos")
    photo_tables = []
    for label in photo_caption_labels:
        photo_tables.append(pl.scan_parquet(f"/home/shared/yelp/photo_table_{label}.parquet"))
    logger.info(f"Reviews_df.shape: {review_df.select(pl.len()).collect().item()}")
    logger.info("Reading Business")
    business_df = pl.scan_parquet(business_table_output_path)
    # pdb.set_trace()

    business_list = caption_tables + photo_tables + [business_df]
    dataframes = [review_df, user_df, business_list]
    names = ["review", "user", "business"]
    idx_cols = [["review_id", "user_id", "business_id", "timestamp", "stars"], ["user_id"], ["business_id"]]

    logger.info("Exploding dfs")
    output_dfs = []
    for dfs, idx_col_list, name in zip(dataframes, idx_cols, names):
        output_path = Path(f"/home/shared/yelp/long_{name}_df.parquet")
        if not output_path.exists():
            logger.info(f"Exploding {name} df")
            if isinstance(dfs, list):
                long_dfs = []
                for df in dfs:
                    long_df: pl.LazyFrame = generate_long_df_explode(
                        df, id_column=idx_col_list[0], timestamp_column=None, num_idx_cols=1
                    )
                    long_dfs.append(long_df)
                pl.concat(long_dfs, how="vertical").sink_parquet(output_path)

            else:
                if "timestamp" in dfs.collect_schema().names():
                    timestamp_column = "timestamp"
                else:
                    timestamp_column = None
                num_idx_cols = len(idx_col_list)
                if len(idx_col_list) > 1:
                    more_ids = True
                long_df: pl.LazyFrame = generate_long_df_explode(
                    dfs,
                    id_column=idx_col_list[0],
                    timestamp_column=timestamp_column,
                    num_idx_cols=num_idx_cols,
                    more_ids=more_ids,
                )
                long_df.sink_parquet(output_path)
        assert os.path.exists(output_path)
        logger.info(f"Loading {output_path} df")
        output_dfs.append(pl.scan_parquet(output_path))

    reveiw_df, user_df, business_df = output_dfs
    logger.info("Grouping business_ids")
    business_df = business_df.group_by(["business_id", "code", "index"]).mean()
    logger.info("Joining dfs")
    reveiw_df, user_df, business_df = review_df.collect(), user_df.collect(), business_df.collect()
    # pivot the user dataframe so we have one row per user
    user_df = user_df.select("user_id", "code", "numerical_value").pivot(
        index="user_id", on="code", values="numerical_value"
    )
    # TODO:pivot the business dataframe so we have one row per business

    import pdb

    pdb.set_trace()

    merged_review_df = review_df.join(user_df, on="user_id", how="left")
    merged_review_df = merged_review_df.join(business_df, on="business_id", how="left")
    merged_review_df.sink_parquet("/home/shared/yelp/merged_review_df.parquet")

    logger.info("Joining dfs")

    import pdb

    pdb.set_trace()
    big_df = long_dfs[0]
    # import pdb; pdb.set_trace()
    # for i in range(1, len(long_dfs)):
    #     if "user_id" in idx_cols[i]:
    #         other = long_dfs[i].join(review_df.select(["review_id", "user_id"]), on="user_id", how="left")
    #     else:
    #         other = review_df.join(long_dfs[i], on="business_id", how="left")
    #     new_df = other.select([
    #             'review_id',
    #             'user_id',
    #             'business_id',
    #             'timestamp',
    #             pl.col('code_right').alias('code'),
    #             pl.col('index_right').alias('index'),
    #             pl.col('numerical_value_right').alias('numerical_value')
    #             ])
    #     big_df = pl.concat([big_df, new_df], how='vertical')
    # logger.info("Grouping dfs")
    # big_df = big_df.group_by(["review_id", "user_id", "business_id", "timestamp", "code", "index"]).mean()
    # logger.info("Sinking/collecting dfs")
    # assert isinstance(big_df, pl.LazyFrame)
    # big_df.sink_parquet("/home/shared/yelp/big_table.parquet")

    # big_df.sink_parquet("/home/shared/yelp/big_table.parquet")
    # import pdb; pdb.set_trace()
    # long_dfs[0].columns == ["review_id", "user_id", "business_id", "code", "timestamp", "numerical_value"]
    # long_dfs[1].columns == ["business_id", "code", "timestamp", "numerical_value"]
    # -> group_by("business_id", "code", "embedding_index").mean()

    # -> left join review_id -- problem is this explodes the size of the dataset

    # long_dfs[0].columns == ["review_id", "user_id", "business_id", "code", "timestamp", "numerical_value"]
    # long_dfs[1].columns == ["review_id", "business_id", "code", "timestamp", "numerical_value"]

    # -> vstack
    # pl.concat(long_dfs, how='vertical')

    # pl.concat(long_dfs)

    # logger.info(f"Reviews_df.shape: {review_df.select(pl.len()).collect().item()}")
    # logger.info("Merging users")
    # review_df = review_df.join(user_df, on="user_id", how="left")
    # logger.info(f"Reviews_df.shape: {review_df.select(pl.len()).collect().item()}")
    # logger.info("Merging captions")
    # for table in caption_tables:
    #     import pdb; pdb.set_trace()
    #     review_df = review_df.join(table, on="business_id", how="left")
    #     logger.info(f"Reviews_df.shape: {review_df.select(pl.len()).collect().item()}")
    # logger.info("Merging photos")
    # for table in photo_tables:
    #     review_df = review_df.join(table, on="business_id", how="left")
    # logger.info(f"Reviews_df.shape: {review_df.select(pl.len()).collect().item()}")
    # logger.info("Merging business")
    # review_df = review_df.join(business_df, on="business_id", how="left")
    # logger.info(f"Reviews_df.shape: {review_df.select(pl.len()).collect().item()}")
    # logger.info("Aggregating")
    # # TODO Fix this aggregation
    # # import pdb; pdb.set_trace()
    # long_df = generate_long_df_explode(review_df, id_column="review_id", timestamp_column="timestamp")
    # # long_df = long_df.group_by("review_id", "code").mean()
    # logger.info("writing")
    # # long_df.sink_parquet("/home/shared/yelp/big_table.parquet")
    # long_df.sink_parquet("/home/shared/yelp/big_table.parquet")


# large_df.select([pl.arange(0, pl.col("category_embeddings").arr.rank().list.len())])


# large_df.select(pl.col("category_embeddings").arr.to_list().eval(pl.element().alias("embedding"), parallel=True).with_columns(pl.arange(0, pl.count()).alias("index")))

# large_df.select([pl.arange(0, pl.col("category_embeddings").arr.to_list().len()).alias("index")])

# large_df.select([pl.col("category_embeddings").explode().alias("embedding"),pl.arange(0, pl.col("category_embeddings").len()).alias("index")])
# large_df.select([pl.col("category_embeddings").explode()])


if __name__ == "__main__":
    # process_photo_captions()
    # process_review()
    # process_business()
    # process_user()
    aggregate_df()
