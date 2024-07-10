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


photo_output_path = "photos.npy"
if not os.path.exists(photo_output_path):
    photo_ids_list = photo_ids
    photo_dataloader = DataLoader(PhotoDataset(photo_ids_list), shuffle=False, batch_size=64, num_workers=8)
    photo_all_embeddings = []
    for batch in tqdm(photo_dataloader):
        batch.to(device)
        with torch.no_grad():
            embeddings = model.get_image_features(pixel_values=batch["pixel_values"].squeeze(dim=1))
            photo_all_embeddings.append(embeddings.cpu().numpy())
    photo_stacked_embeddings = np.vstack(photo_all_embeddings)
    np.save(photo_output_path, photo_stacked_embeddings)


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


caption_output_path = "caption.npy"
if not os.path.exists(caption_output_path):
    captions_dataloader = DataLoader(CaptionDataset(captions), shuffle=False, batch_size=64)
    caption_all_embeddings = []
    for batch in captions_dataloader:
        tokens = tokenizer(batch, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
        tokens = {key: value.to(model.device) for key, value in tokens.items()}

        with torch.no_grad():
            embeddings = model.get_text_features(**tokens)
            caption_all_embeddings.append(embeddings.cpu().numpy())
    caption_stacked_embeddings = np.vstack(caption_all_embeddings)
    np.save(caption_output_path, caption_stacked_embeddings)

# Dataframe of the caption embeddings and the labels
photo_caption_table = {
    "business_ids": caption_business_ids,
    "captions": np.load("caption.npy"),
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

# Dataframe of the image embeddings and the labels
photo_table = {
    "business_ids": photo_business_ids,
    "image_embeddings": np.load("photos.npy"),
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
# print(photo_table)

# Load all of the reviews into a dataframe
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
        stars.append(data["stars"])


def process_review():
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

    review_output_path = "reviews.npy"
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
        "timestamp": np.array(timestamps, dtype="datetime64"),
        "business_ids": review_business,
        "review_embeddings": np.load("reviews.npy"),
        "useful": useful_count,
        "funny": funny_count,
        "cool": cool_count,
        "stars": stars,
    }
    review_table = pl.DataFrame(review_table)


# dataframes = []
# for filepath in filepaths:
#     with open(filepath, 'r') as f:
#         data_list = []
#         for line in f:
#             data = json.loads(line)
#             data_list.append(data)
#             df = pl.DataFrame(data_list)
#             if filepath == "/storage/shared/yelp_dataset_image/photos.json":
#                 image_embeddings = []
#                 text_embeddings = []
#                 for row in df.iter_rows:
#                     image = Image.open("/storage/shared/yelp_dataset_image" +
# "/photos/" + row["photo_id"] + ".jpg")
#                     inputs = processor(text=[row["caption"], row["label"]],
# images=image, return_tensors="pt", padding=True)
#                     outputs = model(**inputs)
#                     model.get_image_features()
#                     model.get_text_features()
#                     image_embeddings.append(outputs.image_embeds)
#                     text_embeddings.append(outputs.text_embeds)
#                 df.with_columns(
#                     img_embeds = pl.Series(image_embeddings),
#                     text_embeds = pl.Series(text_embeddings)
#                     )
#             dataframes.append(df)

# all_df = pl.concat(dataframes, how="diagonal")


# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# image = Image.open("/storage/shared/yelp_dataset_image" + "/photos/" + "zsvj7vloL4L5jhYyPIuVwg" + ".jpg")
# inputs = processor(text=["Nice rock artwork everyw
# ere and craploads of taps.", "inside"], images=image, return_tensors="pt", padding=True)
# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image
# probs = logits_per_image.softmax(dim=1)
# print(logits_per_image)
# image_embeds = outputs.image_embeds
# text_embeds = outputs.text_embeds
# print(image_embeds)
# print(text_embeds)
