# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
import json
import shutil

import pandas as pd
from dotenv import find_dotenv, load_dotenv


def main(input_dir, output_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    annotations_df = pd.read_csv(input_dir / "annotations.csv")
    image_filenames = annotations_df[annotations_df.notna().all(axis=1)]["image"].tolist()
    n = len(image_filenames)
    train_n = int(n * 0.7)
    val_n = int(n * 0.15)

    train_filenames = image_filenames[:train_n]
    val_filenames = image_filenames[train_n:train_n + val_n]
    test_filenames = image_filenames[train_n + val_n:]

    os.makedirs(output_dir / "train", exist_ok=True)
    os.makedirs(output_dir / "val", exist_ok=True)
    os.makedirs(output_dir / "test", exist_ok=True)

    for filenames, split in zip((train_filenames, val_filenames, test_filenames), ("train", "val", "test")):
        logger.info(f"making {split} dataset")
        split_captions = []
        for filename in filenames:
            if not (output_dir / split / filename).is_file():
                shutil.copyfile(input_dir / "images-resized" / filename, output_dir / split / filename)
            try:
                row = annotations_df[annotations_df["image"] == filename]
                caption_d = {
                    "file_name": filename,
                    "text": row["caption"].iloc[0]
                }
                split_captions.append(caption_d)
            except:
                print(filename)
        with (output_dir / split / "metadata.jsonl").open("w") as f:
            for item in split_captions:
                f.write(json.dumps(item) + "\n")
        logger.info(f"{split} dataset created")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    raw_data_dir = project_dir / "data" / "raw"
    processed_data_dir = project_dir / "data" / "processed"

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(raw_data_dir, processed_data_dir)
