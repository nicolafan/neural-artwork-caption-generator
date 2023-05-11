import logging
import os
from pathlib import Path

import joblib
import numpy as np
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from src.utils.dirutils import get_data_dir


def enclip_texts(processor, model, df_captions):
    embeddings = np.empty((1, 512))
    filenames = df_captions["image"].values
    texts = df_captions["caption"].tolist()

    for i in tqdm(range(0, len(texts), 8)):
        batch_texts = texts[i : min(i + 8, len(texts))]
        batch_texts = [text.replace("The artwork depicts ", "") for text in batch_texts]
        batch_texts = [" ".join(text.split(" ")[:40]) for text in batch_texts]

        inputs = processor(
            text=batch_texts, images=None, return_tensors="pt", padding=True
        )

        text_features = model.get_text_features(**inputs)
        embeddings = np.concatenate(
            (embeddings, text_features.detach().numpy()), axis=0
        )

    embeddings = embeddings[1:]
    embeddings_dict = {
        filename: embedding for filename, embedding in zip(filenames, embeddings)
    }
    return embeddings_dict


def enclip_images(processor, model, df_captions, images_dir):
    embeddings = np.empty((1, 512))
    filenames = df_captions["image"].values

    for i in tqdm(range(0, len(filenames), 4)):
        batch_filenames = filenames[i : min(i + 4, len(filenames))]
        batch_images = []
        for filename in batch_filenames:
            image = Image.open(images_dir / filename)
            batch_images.append(image)
        inputs = processor(
            text=None, images=batch_images, return_tensors="pt", padding=True
        )
        images_features = model.get_image_features(inputs["pixel_values"])
        embeddings = np.concatenate(
            (embeddings, images_features.detach().numpy()), axis=0
        )

    embeddings = embeddings[1:]
    embeddings_dict = {
        filename: embedding for filename, embedding in zip(filenames, embeddings)
    }
    return embeddings_dict


def fenclip_images(df_captions, images_dir):
    embeddings = np.load(
        get_data_dir() / "processed" / "clip" / "image_embeddings_alpha_order.npy"
    )
    filenames = sorted(os.listdir(images_dir))
    embeddings = embeddings[1:]
    embeddings_dict = {}

    for i, filename in enumerate(filenames):
        embeddings_dict[filename] = embeddings[i]

    return embeddings_dict


def main(input_dir, output_dir):
    """Creates CLIP embeddings for images in images_dir and saves them to
    output_dir.

    The order of the embeddings corresponds to the alphabetical order of the filenames
    returned by os.listdir(images_dir).

    Args:
        images_dir (Path): Path to directory containing images.
        output_dir (Path): Path to directory where CLIP embeddings will be saved.
    """
    logger = logging.getLogger(__name__)

    filenames = sorted(os.listdir(input_dir / "images"))
    filenames_lookup = {filename: i for i, filename in enumerate(filenames)}

    df_captions = pd.read_csv(input_dir / "artgraph_captions.csv")
    df_captions["filename_index"] = df_captions["image"].apply(
        lambda x: filenames_lookup[x]
    )
    df_captions.sort_values(by="filename_index", inplace=True)

    model_url = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_url)
    processor = CLIPProcessor.from_pretrained(model_url)

    logger.info(f"creating CLIP embeddings for images in {input_dir / 'images'}")
    img_embeddings_dict = enclip_images(df_captions, input_dir / "images")
    logger.info(
        f"creating CLIP embeddings for texts in {input_dir / 'artgraph_captions.csv'}"
    )
    caption_embeddings_dict = enclip_texts(processor, model, df_captions)

    embeddings_dict = {}
    for filename, img_embedding in img_embeddings_dict.items():
        caption_embedding = caption_embeddings_dict[filename]
        embeddings_dict[filename] = {
            "img_embedding": img_embedding,
            "caption_embedding": caption_embedding,
        }

    clip_embeddings_path = output_dir / "clip" / "dataset_embeddings.joblib"
    joblib.dump(embeddings_dict, clip_embeddings_path)
    logger.info(f"CLIP embeddings created at {clip_embeddings_path}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    data_dir = get_data_dir()
    raw_data_dir = data_dir / "raw"
    processed_data_dir = data_dir / "processed"

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(raw_data_dir, processed_data_dir)
