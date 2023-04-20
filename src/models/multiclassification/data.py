import numpy as np
from datasets import load_from_disk
from joblib import load
from sklearn.utils.class_weight import compute_class_weight
from transformers import ViTImageProcessor
import torch

from src.utils.dirutils import get_data_dir

MULTICLASS_FEATURES = ("artist", "style", "genre")
MULTILABEL_FEATURES = ("tags", "media")


def get_dataset_for_multiclassification():
    """Returns the processed dataset for multiclassification."""
    dataset = load_from_disk(get_data_dir() / "processed" / "multiclassification_dataset")
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    def process(examples):
        examples["pixel_values"] = processor(examples["image"], return_tensors="pt").pixel_values
        return examples

    dataset = dataset.map(process, remove_columns="image", batched=True)
    return dataset


def get_multiclassification_dicts():
    multiclass_classifications = {}
    multilabel_classifications = {}

    for feature in MULTICLASS_FEATURES:
        ordinal_encoder = load(get_data_dir() / "processed" / "ordinal_encoders" / f"{feature}.joblib")
        multiclass_classifications[feature] = len(ordinal_encoder.categories_[0])

    for feature in MULTILABEL_FEATURES:
        multilabel_binarizer = load(get_data_dir() / "processed" / "multilabel_binarizers" / f"{feature}.joblib")
        multilabel_classifications[feature] = len(multilabel_binarizer.classes_)

    return multiclass_classifications, multilabel_classifications


def compute_class_weight_tensors(dataset, device):
    class_weights = {}
    for feature in MULTICLASS_FEATURES:
        y_org = np.unique(dataset["train"][feature])
        # remove -1 from y_org
        y_org = y_org[y_org != -1]
        y = [item for sublist in dataset["train"][feature] for item in sublist]
        y = [item for item in y if item != -1]
        class_weights[feature] = torch.tensor(compute_class_weight("balanced", classes=y_org, y=y)).float().to(device)
    return class_weights
