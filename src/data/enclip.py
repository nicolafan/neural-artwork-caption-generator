import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from transformers import CLIPModel, CLIPProcessor
import numpy as np
import os
from tqdm import tqdm
from PIL import Image


def main(images_dir, processed_data_dir):
    logger = logging.getLogger(__name__)
    logger.info(f"creating CLIP embeddings for images in {images_dir}")

    model_url = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_url)
    processor = CLIPProcessor.from_pretrained(model_url)

    embeddings = np.empty((1, 512))
    filenames = sorted(os.listdir(images_dir))

    for i in tqdm(range(0, len(filenames), 4)):
        batch_filenames = filenames[i:min(i+4, len(filenames))]
        batch_images = []
        for filename in batch_filenames:
            image = Image.open(images_dir / filename)
            batch_images.append(image)
        inputs = processor(text=None, images=batch_images, return_tensors="pt", padding=True)
        images_features = model.get_image_features(inputs["pixel_values"])
        embeddings = np.concatenate((embeddings, images_features.detach().numpy()), axis=0)

    clip_embeddings_path = processed_data_dir / "clip_embeddings.npy"
    np.save(clip_embeddings_path, embeddings)
    logger.info(f"CLIP embeddings created at {clip_embeddings_path}!")



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    images_dir = project_dir / "data" / "raw" / "images"
    processed_data_dir = project_dir / "data" / "processed"

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(images_dir, processed_data_dir)