import os
from pathlib import Path
from datasets import load_dataset
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, MultiLabelBinarizer
from functools import partial


def _encode_multiclass_feature(examples, feature, encoder):
    examples[feature] = encoder.transform(np.array(examples[feature]).reshape(-1, 1))
    return examples


def _binarize_multilabel_feature(examples, feature, binarizer):
    examples[feature] = binarizer.transform(
        [[] if labels is None else labels.split(", ") for labels in examples[feature]]
    )
    return examples


def _get_fitted_ordinal_encoder(dataset, feature):
    oe = OrdinalEncoder(
        categories="auto", dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1
    )
    categories = [category for category in dataset["train"][feature] if category is not None]
    training_values = np.array(categories).reshape(-1, 1)
    return oe.fit(training_values)


def _get_fitted_multilabel_binarizer(dataset, feature):
    mlb = MultiLabelBinarizer()
    training_values = [
        [] if labels is None else labels.split(", ")
        for labels in dataset["train"][feature]
    ]
    return mlb.fit(training_values)


def get_prepared_dataset_for_multiclassification(data_dir):
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
            partial(_encode_multiclass_feature, feature=feature, encoder=encoder), batched=True
        )

    for feature, binarizer in multilabel_binarizers.items():
        dataset = dataset.map(
            partial(_binarize_multilabel_feature, feature=feature, binarizer=binarizer), batched=True
        )

    dataset = dataset.remove_columns(["caption", "human"])
    return dataset, ordinal_encoders, multilabel_binarizers
