import logging
import os
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm


def _get_valid_labels(df, col, min_label_count):
    """Get all labels that appear in multilabel df col at least min_label_count times

    Args:
        df (pd.DataFrame): Dataframe
        col (str): Column name
        min_label_count (int): Min number of appearances for valid labels

    Returns:
        set: Valid labels
    """
    multilabels = df[col].tolist()
    multilabels = [multilabel for multilabel in multilabels if type(multilabel) == str]
    labels = []
    for multilabel in multilabels:
        labels += multilabel.split(", ")

    freqs = Counter(labels)
    valid_labels = [freq[0] for freq in freqs.items() if freq[1] >= min_label_count]

    return set(valid_labels)


def clean_dataset(df, min_label_count):
    """Clean the dataframe

    Args:
        df (_type_): _description_
        min_label_count (_type_): _description_

    Returns:
        _type_: _description_
    """
    # set to NA artists with less than min_label_count appearances
    artists_df = df.groupby(["artist"]).count().reset_index()
    valid_artists = set(
        artists_df[artists_df["image"] >= min_label_count]["artist"].values
    )
    df.loc[df["artist"].isin(valid_artists), "artist"] = pd.NA

    def keep_valid_labels(x, valid_labels):
        if not pd.notnull(x):
            return pd.NA
        labels = x.split(", ")
        labels = [label for label in labels if label in valid_labels]
        if labels:
            return ", ".join(labels)
        else:
            return pd.NA

    # set to NA tags with less than min_label_count appearances
    valid_tags = _get_valid_labels(df, "tags", min_label_count)
    df["tags"] = df["tags"].apply(keep_valid_labels, valid_labels=valid_tags)

    # set to NA media with less than min_label_count appearances
    valid_media = _get_valid_labels(df, "media", min_label_count)
    df["media"] = df["media"].apply(keep_valid_labels, valid_labels=valid_media)

    # save clean dataframe to csv
    return df


def main(min_label_count, input_dir, output_dir):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making classification data set from raw data")

    # read and clean the dataset
    df = pd.read_csv(input_dir / "artgraph_dataset.csv")
    df = clean_dataset(df, min_label_count)

    # read the caption dataset
    captions_df = pd.read_csv(input_dir / "artgraph_captions.csv")

    # merge the datasets
    df = df.merge(captions_df, how="left", on="image")
    df = df.drop(columns="name")

    logger.info("full dataset created")

    # create splits
    images_dir = input_dir / "images"
    splits = ("train", "test", "val")
    for split in splits:
        # read splits and make split dirs
        with (input_dir / "splits" / f"{split}.txt").open("r") as f:
            filenames = [line.strip() for line in f.readlines()]
        try:
            split_dir = output_dir / split
            os.makedirs(split_dir)
        except:
            logger.error(
                f"can't create the {split} split directory, delete it if it already exists"
            )
        
        logger.info(f"copying images for {split} split")
        for filename in tqdm(filenames):
            shutil.copy(images_dir / filename, split_dir / filename)
        logger.info("images copied")

        logger.info("making metadata")
        split_df = df[df["image"].isin(filenames)]
        split_df = split_df.rename({"image": "file_name"})

        split_df.to_csv(split_dir / "metadata.csv", index=False)
        logger.info(f"{split} set created")

    logger.info(f"dataset created at {output_dir}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    raw_data_dir = project_dir / "data" / "raw"
    processed_data_dir = project_dir / "data" / "processed"

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(100, raw_data_dir, processed_data_dir)
