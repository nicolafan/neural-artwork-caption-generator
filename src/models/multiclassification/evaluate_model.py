import math

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, hamming_loss,
                             precision_score, recall_score)
from tqdm import tqdm

import src.models.multiclassification.data as data
from src.models.multiclassification.losses import join_losses, losses_fn


def _concatenate_batch_arrays(all_data):
    """Concatenates arrays in a batch.

    Args:
        all_data (list[dict]): list of dictionaries containing arrays

    Returns:
        dict: dictionary of concatenated arrays
    """
    all_data_concat = {}
    for d in all_data:
        for key in d:
            if key not in all_data_concat:
                all_data_concat[key] = d[key]
            else:
                all_data_concat[key] = np.concatenate((all_data_concat[key], d[key]), axis=0)
    return all_data_concat


def compute_metrics(outputs, targets, multiclass_features, multilabel_features):
    """Computes metrics for multi-classification and multi-label classification.

    Args:
        outputs (dict): dictionary of outputs
        targets (dict): dictionary of targets
        multiclass_features (list[str]): list of multiclass features
        multilabel_features (list[str]): list of multilabel features

    Returns:
        dict: dictionary of metrics
    """
    for feature in multiclass_features:
        outputs[feature] = outputs[feature].argmax(-1)
    for feature in multilabel_features:
        outputs[feature] = np.where(outputs[feature] > 0, 1, 0)

    metrics = {}

    # Computing accuracy, precision, recall, and F1 for multiclass classifications
    for feature in multiclass_features:
        # if feature == "artist" mask also 174
        mask = targets[feature] != -1 # remove invalid predictions (if present)
        targets[feature] = np.squeeze(targets[feature][mask])
        outputs[feature] = outputs[feature][:, np.newaxis]
        outputs[feature] = np.squeeze(outputs[feature][mask])

        accuracy = accuracy_score(targets[feature], outputs[feature])
        precision = precision_score(
            targets[feature], outputs[feature], average="macro", zero_division=0
        )
        recall = recall_score(
            targets[feature], outputs[feature], average="macro", zero_division=0
        )
        f1 = f1_score(
            targets[feature], outputs[feature], average="macro", zero_division=0
        )
        metrics[f"{feature}_accuracy"] = accuracy
        metrics[f"{feature}_macro_precision"] = precision
        metrics[f"{feature}_macro_recall"] = recall
        metrics[f"{feature}_macro_f1"] = f1

    # Computing hamming loss, precision, recall, and F1 for multilabel classifications
    for feature in multilabel_features:
        row_sum = np.sum(targets[feature], axis=-1)
        mask = row_sum != 0
        targets[feature] = targets[feature][mask]
        outputs[feature] = outputs[feature][mask]

        hamming = hamming_loss(targets[feature], outputs[feature])
        precision = precision_score(
            targets[feature], outputs[feature], average="macro", zero_division=0
        )
        recall = recall_score(
            targets[feature], outputs[feature], average="macro", zero_division=0
        )
        f1 = f1_score(
            targets[feature], outputs[feature], average="macro", zero_division=0
        )
        metrics[f"{feature}_hamming_loss"] = hamming
        metrics[f"{feature}_macro_precision"] = precision
        metrics[f"{feature}_macro_recall"] = recall
        metrics[f"{feature}_macro_f1"] = f1

    avg_macro_f1 = 0
    for feature in multiclass_features + multilabel_features:
        avg_macro_f1 += metrics[f"{feature}_macro_f1"]
    avg_macro_f1 /= len(multiclass_features) + len(multilabel_features)
    metrics["avg_macro_f1"] = avg_macro_f1

    return metrics


@torch.no_grad()
def evaluate(model, dataloader, class_weight_tensors, multiclass_features, multilabel_features, batch_size, num_accumulation_steps):
    """Evaluates a model.

    Args:
        model (ViTForMultiClassification): model to evaluate
        dataloader (DataLoader): dataloader to use
        class_weight_tensors (dict): dictionary of class weight tensors
        multiclass_features (list[str]): list of multiclass features
        multilabel_features (list[str]): list of multilabel features
        batch_size (int): batch size
        num_accumulation_steps (int): number of accumulation steps

    Returns:
        tuple: tuple of (average loss, average label losses, dictionary of metrics)
    """
    all_features = multiclass_features + multilabel_features

    # Validate
    running_vloss = 0.
    running_label_vlosses = [0.] * len(all_features)
    all_voutputs = []
    all_vtargets = []
    n_batches = math.ceil(len(dataloader) * batch_size / 32) # virtual number of batches
    
    for vbatch in tqdm(dataloader):
        # Compute batch outputs
        vinputs, vtargets = vbatch["pixel_values"], {k: vbatch[k] for k in vbatch if k != "pixel_values"}
        voutputs = model(vinputs)

        # Compute batch losses
        vlosses = losses_fn(multiclass_features, multilabel_features, voutputs, vtargets, class_weight_tensors)
        vloss = join_losses(model, vlosses)
        for i, _ in enumerate(all_features):
            running_label_vlosses[i] += vlosses[i].item() / num_accumulation_steps
        running_vloss += vloss.item() / num_accumulation_steps

        # Save predictions and targets for later
        all_voutputs.append(
            {k: v.detach().cpu().numpy() for k, v in voutputs.items()}
        )
        all_vtargets.append(
            {k: v.detach().cpu().numpy() for k, v in vtargets.items()}
        )

    avg_vloss = running_vloss / n_batches
    running_label_vlosses = [loss / n_batches for loss in running_label_vlosses]
    all_voutputs = _concatenate_batch_arrays(all_voutputs)
    all_vtargets = _concatenate_batch_arrays(all_vtargets)
    metrics = compute_metrics(all_voutputs, all_vtargets, multiclass_features, multilabel_features)
    return avg_vloss, running_label_vlosses, metrics