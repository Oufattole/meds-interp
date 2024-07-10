import json
import os

import numpy as np
import polars as pl
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = processor.tokenizer
image_processor = processor.image_processor

device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda"
model.to(device)

filepaths = [
    "/storage/shared/yelp_dataset/raw/yelp_academic_dataset_business.json",
    "/storage/shared/yelp_dataset/raw/yelp_academic_dataset_user.json",
    "/storage/shared/yelp_dataset/raw/yelp_academic_dataset_review.json",
    "/storage/shared/yelp_dataset_image/photos.json",
]


with open("/storage/shared/yelp_dataset_image/photos.json") as r:
    photo_ids = []
    photo_business_ids = []
    photo_labels = []
    captions = []
    caption_labels = []
    caption_business_ids = []
    for line in r:
        data = json.loads(line)
        try:
            img_path = "/storage/shared/yelp_dataset_image" + "/photos/" + data["photo_id"] + ".jpg"
            image = Image.open(img_path)
            photo_ids.append(data["photo_id"])
            photo_business_ids.append(data["business_id"])
            photo_labels.append(data["label"])
            if data["caption"]:
                captions.append(data["caption"])
                caption_labels.append(data["label"])
                caption_business_ids.append(data["business_id"])
        except UnidentifiedImageError:
            continue


def process_photos():
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
            img_path = "/storage/shared/yelp_dataset_image" + "/photos/" + self.photo_ids_list[idx] + ".jpg"
            image = Image.open(img_path)
            image_tensor = image_processor(image, return_tensors="pt")
            return image_tensor

    photo_output_path = "/home/leander/MIT_projects/meds-interp/photos.npy"
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
                photo_all_embeddings.append(embeddings.cpu().numpy())
        photo_stacked_embeddings = np.vstack(photo_all_embeddings)
        np.save(photo_output_path, photo_stacked_embeddings)

    # Dataframe of the image embeddings and the labels
    photo_table = {
        "business_id": photo_business_ids,
        "image_embeddings": np.load(photo_output_path),
        "labels": photo_labels,
    }
    photo_table = pl.DataFrame(photo_table)
    unique_photo_labels = photo_table["labels"].unique().to_list()
    for label in unique_photo_labels:
        photo_table = photo_table.with_columns(
            pl.when(pl.col("labels") == label)
            .then(pl.col("image_embeddings"))
            .otherwise(None)
            .alias(f"embeddings_{label}")
        )
    photo_table.drop(["captions", "labels"])
    return photo_table


def process_captions():
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

    caption_output_path = "/home/leander/MIT_projects/meds-interp/captions.npy"
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
        "business_ids": caption_business_ids,
        "captions": np.load(caption_output_path),
        "labels": caption_labels,
    }
    photo_caption_table = pl.DataFrame(photo_caption_table)
    unique_caption_labels = photo_caption_table["labels"].unique().to_list()
    for label in unique_caption_labels:
        photo_caption_table = photo_caption_table.with_columns(
            pl.when(pl.col("labels") == label)
            .then(pl.col("captions"))
            .otherwise(None)
            .alias(f"embeddings_{label}")
        )
    photo_caption_table.drop(["captions", "labels"])
    return photo_caption_table


# Load all of the reviews into a dataframe
def process_review():
    with open("/storage/shared/yelp_dataset/raw/yelp_academic_dataset_review.json") as r:
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

    review_output_path = "/home/leander/MIT_projects/meds-interp/reviews.npy"
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

    review_table = {
        "user_id": review_user,
        "timestamp": np.array(timestamps, dtype="datetime64[ns]"),
        "business_ids": review_business,
        "review_embeddings": np.load("/home/leander/MIT_projects/meds-interp/reviews.npy"),
        "useful": useful_count,
        "funny": funny_count,
        "cool": cool_count,
        "stars": stars,
    }
    review_table = pl.DataFrame(review_table)

    return review_table


def process_business():
    with open("/storage/shared/yelp_dataset/raw/yelp_academic_dataset_business.json") as r:
        business_id = []
        stars = []
        review_count = []
        attributes = []
        categories = []
        for line in r:
            data = json.loads(line)
            if isinstance(data["categories"], str):
                business_id.append(data["business_id"])
                stars.append(data["stars"])
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

    business_categories_output_path = "/home/leander/MIT_projects/meds-interp/business_categories.npy"
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

    business_attributes_output_path = "/home/leander/MIT_projects/meds-interp/business_attributes.npy"
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
        "business_id": business_id,
        "attribute_embeddings": np.load(business_attributes_output_path),
        "category_embeddings": np.load(business_categories_output_path),
        "stars": stars,
        "review_count": review_count,
    }
    business_table = pl.DataFrame(business_table)


def process_user():
    with open("/storage/shared/yelp_dataset/raw/yelp_academic_dataset_user.json") as r:
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
        "user_id": user_id,
        "yelping_since": np.array(yelping_since, dtype="datetime64[ns]"),
        "review_count": review_count,
        "useful_sent": useful_sent,
        "funny_sent": funny_sent,
        "cool_sent": cool_sent,
        "fans": fans,
        "num_years_elite": elite,
        "average_stars": average_stars,
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
    # print(review_user_grouped)

    photos_grouped = (
        process_photos()
        .group_by("business_id", maintain_order=True)
        .agg(
            [
                pl.col("embeddings_inside").map_elements(mean_vector, return_dtype=pl.List(pl.Float64)),
                pl.col("embeddings_outside").map_elements(mean_vector, return_dtype=pl.List(pl.Float64)),
                pl.col("embeddings_food").map_elements(mean_vector, return_dtype=pl.List(pl.Float64)),
                pl.col("embeddings_drink").map_elements(mean_vector, return_dtype=pl.List(pl.Float64)),
                pl.col("embeddings_menu").map_elements(mean_vector, return_dtype=pl.List(pl.Float64)),
            ]
        )
    )
    for label in ["inside", "outside", "food", "drink", "menu"]:
        photos_grouped = photos_grouped.with_columns(
            pl.when(pl.col(f"embeddings_{label}") == [])
            .then(None)
            .otherwise(pl.col(f"embeddings_{label}"))
            .alias(f"embeddings_{label}")
        )
    print(photos_grouped)


if __name__ == "__main__":
    # process_captions()
    # process_photos()
    # process_review()
    # process_business()
    # process_user()
    aggregate_df()
