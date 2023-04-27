import logging
import os
from pathlib import Path

import numpy as np
from dotenv import find_dotenv, load_dotenv
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from src.utils.dirutils import get_data_dir


def main(images_dir, output_dir):
    """Creates CLIP embeddings for images in images_dir and saves them to
    output_dir.

    The order of the embeddings corresponds to the alphabetical order of the filenames
    returned by os.listdir(images_dir).
    The first embedding must be discarded as it is a dummy embedding.

    Args:
        images_dir (Path): Path to directory containing images.
        output_dir (Path): Path to directory where CLIP embeddings will be saved.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"creating CLIP embeddings for images in {images_dir}")

    model_url = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_url)
    processor = CLIPProcessor.from_pretrained(model_url)

    embeddings = np.empty((1, 512))
    filenames = sorted(os.listdir(images_dir))

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

    clip_embeddings_path = output_dir / "clip_embeddings.npy"
    np.save(clip_embeddings_path, embeddings)
    logger.info(f"CLIP embeddings created at {clip_embeddings_path}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    data_dir = get_data_dir()
    images_dir = data_dir / "raw" / "images"
    processed_data_dir = data_dir / "processed"

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(images_dir, processed_data_dir)
