import numpy as np
from datasets import Dataset, load_from_disk
from joblib import load
from sklearn.utils.class_weight import compute_class_weight
from transformers import ViTImageProcessor
import torch

from src.utils.dirutils import get_data_dir

MULTICLASS_FEATURES = ("artist", "style", "genre")
MULTILABEL_FEATURES = ("tags", "media")


def get_multiclassification_dicts():
    """Get multiclassification dicts.

    Returns:
        multiclass_classifications (dict): dict with number of classes for each classification feature
    """
    multiclass_classifications = {}
    multilabel_classifications = {}

    for feature in MULTICLASS_FEATURES:
        ordinal_encoder = load(
            get_data_dir() / "processed" / "ordinal_encoders" / f"{feature}.joblib"
        )
        multiclass_classifications[feature] = len(ordinal_encoder.categories_[0])

    for feature in MULTILABEL_FEATURES:
        multilabel_binarizer = load(
            get_data_dir() / "processed" / "multilabel_binarizers" / f"{feature}.joblib"
        )
        multilabel_classifications[feature] = len(multilabel_binarizer.classes_)

    return multiclass_classifications, multilabel_classifications


def compute_class_weight_tensors(dataset, device):
    """Compute class weight tensors.

    Args:
        dataset (datasets.Dataset): dataset for multiclassification
        device (torch.device): device to put the tensors on
    Returns:
        class_weights (dict): dict with class weight tensors for each classification feature
    """
    class_weights = {}
    for feature in MULTICLASS_FEATURES:
        # check if feature is a column in dataset
        if feature not in dataset["train"].column_names:
            continue
        y_org = np.unique(dataset["train"][feature])
        # remove -1 from y_org
        y_org = y_org[y_org != -1]
        y = [item for sublist in dataset["train"][feature] for item in sublist]
        y = [item for item in y if item != -1]
        class_weights[feature] = (
            torch.tensor(compute_class_weight("balanced", classes=y_org, y=y))
            .float()
            .to(device)
        )
    return class_weights


def transform_for_model(examples, processor, device):
    """Transform examples for model.

    Args:
        examples (dict): dict with examples
        processor (transformers.ViTImageProcessor): image processor
        device (torch.device): device to put the tensors on
    Returns:
        new_examples (dict): dict with transformed examples
    """
    examples["pixel_values"] = processor(examples["image"], return_tensors="pt").pixel_values
    new_examples = {}
    for feature in MULTICLASS_FEATURES:
        new_examples[feature] = torch.tensor(examples[feature], dtype=torch.long, device=device)
    for feature in MULTILABEL_FEATURES:
        new_examples[feature] = torch.tensor(examples[feature], dtype=torch.float, device=device)
    new_examples["pixel_values"] = examples["pixel_values"].to(device)
    return new_examples
