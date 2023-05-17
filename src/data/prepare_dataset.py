import joblib
import os
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder
from src.features.tokenize import SpacyTokenizer


def _encode_multiclass_feature(examples, feature, encoder):
    """Encodes a multiclass feature using a fitted encoder.

    Args:
        examples (dict): input examples
        feature (str): feature to encode
        encoder (sklearn.preprocessing.OrdinalEncoder): fitted encoder

    Returns:
        examples (dict): examples with encoded feature
    """
    examples[feature] = encoder.transform(np.array(examples[feature]).reshape(-1, 1))
    return examples


def _binarize_multilabel_feature(examples, feature, binarizer):
    """Binarizes a multilabel feature using a fitted binarizer.

    Args:
        examples (dict): input examples
        feature (str): feature to binarize
        binarizer (sklearn.preprocessing.MultiLabelBinarizer): fitted binarizer

    Returns:
        examples (dict): examples with binarized feature
    """
    examples[feature] = binarizer.transform(
        [[] if labels is None else labels.split(", ") for labels in examples[feature]]
    )
    return examples


def _get_fitted_ordinal_encoder(dataset, feature):
    """Fits an ordinal encoder to the training data.

    Args:
        dataset (datasets.Dataset): dataset
        feature (str): feature to encode

    Returns:
        oe (sklearn.preprocessing.OrdinalEncoder): fitted encoder
    """
    oe = OrdinalEncoder(
        categories="auto",
        dtype=np.int64,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    categories = [
        category for category in dataset["train"][feature] if category is not None
    ]
    training_values = np.array(categories).reshape(-1, 1)
    return oe.fit(training_values)


def _get_fitted_multilabel_binarizer(dataset, feature):
    """Fits a multilabel binarizer to the training data.

    Args:
        dataset (datasets.Dataset): dataset
        feature (str): feature to binarize

    Returns:
        mlb (sklearn.preprocessing.MultiLabelBinarizer): fitted binarizer
    """
    mlb = MultiLabelBinarizer()
    training_values = [
        [] if labels is None else labels.split(", ")
        for labels in dataset["train"][feature]
    ]
    return mlb.fit(training_values)


def _sort_dataset(dataset, data_dir):
    for split in dataset.keys():
        # read data_dir/metadata.csv
        metadata = pd.read_csv(Path(data_dir) / (split if split != "validation" else "val") / "metadata.csv")
        # get list of locations of file_name in dataset in the metadata column
        metadata_order = {}
        for idx, row in metadata.iterrows():
            metadata_order[row["file_name"]] = idx
        order = [metadata_order[os.path.basename(x["image"].filename)] for x in dataset[split]]

        # sort dataset[split] by filename according to the file_name column in metadata
        dataset[split] = dataset[split].add_column("order", order)
        dataset[split] = dataset[split].sort("order")
        dataset[split] = dataset[split].remove_columns(["order"])
    return dataset


def get_prepared_dataset_for_captioning(data_dir):
    dataset = load_dataset("imagefolder", data_dir=data_dir)

    if (data_dir / "clip").is_dir():
        clip_embeddings_dict = joblib.load(data_dir / "clip" / "dataset_embeddings.joblib")

        def _add_clip_scores(example):
            embeddings = clip_embeddings_dict[os.path.basename(example["image"].filename.replace("@@", ""))]
            score = cosine_similarity(
                [embeddings["img_embedding"]], [embeddings["caption_embedding"]]
            )[0][0]
            if score < 0:
                score = 0
            example["clip_score"] = score
            return example
        
        dataset = dataset.map(_add_clip_scores)

    def _add_filename(example):
        example["file_name"] = os.path.basename(example["image"].filename.replace("@@", ""))
        return example
    dataset = dataset.map(_add_filename)
    dataset = dataset.remove_columns(["artist", "genre", "style", "tags", "media", "human"])
    dataset = _sort_dataset(dataset, data_dir)

    return dataset


def get_prepared_dataset_for_multiclassification(data_dir):
    """Loads the dataset and prepares it for multiclassification.

    Args:
        data_dir (str): path to the data directory

    Returns:
        dataset (datasets.Dataset): dataset
        ordinal_encoders (dict): dictionary of fitted ordinal encoders
        multilabel_binarizers (dict): dictionary of fitted multilabel binarizers
    """
    dataset = load_dataset("imagefolder", data_dir=data_dir)

    multiclass_features = ["artist", "genre", "style"]
    multilabel_features = ["tags", "media"]

    ordinal_encoders = dict(
        (feature, _get_fitted_ordinal_encoder(dataset, feature))
        for feature in multiclass_features
    )
    multilabel_binarizers = dict(
        (feature, _get_fitted_multilabel_binarizer(dataset, feature))
        for feature in multilabel_features
    )

    for feature, encoder in ordinal_encoders.items():
        dataset = dataset.map(
            partial(_encode_multiclass_feature, feature=feature, encoder=encoder),
            batched=True,
        )

    for feature, binarizer in multilabel_binarizers.items():
        dataset = dataset.map(
            partial(_binarize_multilabel_feature, feature=feature, binarizer=binarizer),
            batched=True,
        )

    def _add_filename(example):
        example["file_name"] = os.path.basename(example["image"].filename)
        return example
    dataset = dataset.map(_add_filename)
    dataset = dataset.remove_columns(["caption", "human"])
    dataset = _sort_dataset(dataset, data_dir)
    
    return dataset, ordinal_encoders, multilabel_binarizers
