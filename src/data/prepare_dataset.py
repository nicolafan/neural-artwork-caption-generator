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
    examples[feature] = binarizer.transform([[] if tags is None else tags.split(", ") for tags in examples[feature]])
    return examples


def _get_fitted_ordinal_encoder(dataset, feature):
    oe = OrdinalEncoder(categories="auto", handle_unknown="use_encoded_value", unknown_value=-1)
    training_values = dataset["train"][feature].reshape(-1, 1)
    return oe.fit(training_values)


def _get_fitted_multilabel_binarizer(dataset, feature):
    mlb = MultiLabelBinarizer()
    training_values = dataset["train"][feature]
    return mlb.fit(training_values)


def get_prepared_dataset_for_multiclassification(data_dir):
    dataset = load_dataset("imagefolder", data_dir=data_dir)

    multiclass_features = ("artist", "genre", "style")
    multilabel_features = ("tags", "media")

    ordinal_encoders = dict((feature, _get_fitted_ordinal_encoder(dataset, feature)) for feature in multiclass_features)
    multilabel_binarizers = dict((feature, _get_fitted_multilabel_binarizer(dataset, feature)) for feature in multilabel_features)

    for feature, encoder in ordinal_encoders:
        dataset = dataset.map(partial(_encode_multiclass_feature, feature=feature, encoder=encoder))

    for feature, binarizer in multilabel_binarizers:
        dataset = dataset.map(partial(_binarize_multilabel_feature, feature=feature, binarizer=binarizer))

    dataset = dataset.remove_columns(["caption", "human"])
    return dataset






