import os
from pathlib import Path
from datasets import load_dataset
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, MultiLabelBinarizer
from functools import partial


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

    dataset = dataset.remove_columns(["caption", "human"])
    return dataset, ordinal_encoders, multilabel_binarizers
