import json

import numpy as np
import polars as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = processor.tokenizer
image_processor = processor.image_processor

filepaths = [
    "/storage/shared/yelp_dataset/raw/yelp_academic_dataset_business.json",
    "/storage/shared/yelp_dataset/raw/yelp_academic_dataset_user.json",
    "/storage/shared/yelp_dataset/raw/yelp_academic_dataset_review.json",
    "/storage/shared/yelp_dataset_image/photos.json",
]


with open("/storage/shared/yelp_dataset_image/photos.json") as r:
    photo_ids = []
    captions = []
    labels = []
    valid_business_ids = []
    for line in r:
        data = json.loads(line)
        photo_ids.append(data["photo_id"])
        if data["caption"]:
            captions.append(data["caption"])
            labels.append(data["label"])

device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda"
model.to(device)


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


photo_ids_list = photo_ids
photo_dataloader = DataLoader(PhotoDataset(photo_ids_list), shuffle=False, batch_size=64, num_workers=8)
photo_all_embeddings = []
for batch in tqdm(photo_dataloader):
    batch.to(device)
    with torch.no_grad():
        embeddings = model.get_image_features(pixel_values=batch["pixel_values"].squeeze(dim=1))
        photo_all_embeddings.append(embeddings.cpu().numpy())
photo_stacked_embeddings = np.vstack(photo_all_embeddings)


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


captions_dataloader = DataLoader(CaptionDataset(captions, shuffle=False, batch_size=64))
caption_all_embeddings = []
for batch in captions_dataloader:
    batch.to(device)
    with torch.no_grad():
        embeddings = model.get_text_features(batch)
        caption_all_embeddings.append(embeddings.cpu().to_numpy())
caption_stacked_embeddings = np.vstack(caption_all_embeddings)

photo_caption_table = {
    "business_ids": valid_business_ids,
    "captions": caption_stacked_embeddings,
    "labels": labels,
}
photo_caption_table = pl.DataFrame(photo_caption_table)


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
