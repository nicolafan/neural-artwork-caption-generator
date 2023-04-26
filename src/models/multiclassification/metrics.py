import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    hamming_loss,
)
import src.models.multiclassification.data as data


def compute_metrics(eval_pred):
    """Computes metrics for multi-classification and multi-label classification.

    Args:
        eval_pred (transformers.EvalPrediction):

    Returns:
        dict: dictionary of metrics
    """
    predictions, labels = eval_pred
    multiclass_logits, multilabel_logits = list(
        predictions[: len(data.MULTICLASS_FEATURES)]
    ), list(predictions[len(data.MULTICLASS_FEATURES) :])
    multiclass_labels, multilabel_labels = list(
        labels[: len(data.MULTICLASS_FEATURES)]
    ), list(labels[len(data.MULTICLASS_FEATURES) :])

    multiclass_preds = [logits.argmax(-1) for logits in multiclass_logits]
    multilabel_preds = []
    for logits in multilabel_logits:
        pred = logits.copy()
        pred[logits < 0] = 0
        pred[logits > 0] = 1
        multilabel_preds.append(pred)

    metrics = {}

    # Computing accuracy, precision, recall, and F1 for multiclass classifications
    for i, feature in enumerate(data.MULTICLASS_FEATURES):
        mask = multiclass_labels[i] != -1
        multiclass_labels[i] = np.squeeze(multiclass_labels[i][mask])
        multiclass_preds[i] = multiclass_preds[i][:, np.newaxis]
        multiclass_preds[i] = np.squeeze(multiclass_preds[i][mask])

        accuracy = accuracy_score(multiclass_labels[i], multiclass_preds[i])
        precision = precision_score(
            multiclass_labels[i], multiclass_preds[i], average="macro", zero_division=0
        )
        recall = recall_score(
            multiclass_labels[i], multiclass_preds[i], average="macro", zero_division=0
        )
        f1 = f1_score(
            multiclass_labels[i], multiclass_preds[i], average="macro", zero_division=0
        )
        metrics[f"{feature}_accuracy"] = accuracy
        metrics[f"{feature}_macro_precision"] = precision
        metrics[f"{feature}_macro_recall"] = recall
        metrics[f"{feature}_macro_f1"] = f1

    # Computing hamming loss, precision, recall, and F1 for multilabel classifications
    for i, feature in enumerate(data.MULTILABEL_FEATURES):
        row_sum = np.sum(multilabel_labels[i], axis=-1)
        mask = row_sum != 0
        multilabel_labels[i] = multilabel_labels[i][mask]
        multilabel_preds[i] = multilabel_preds[i][mask]

        try:
            hamming = hamming_loss(multilabel_labels[i], multilabel_preds[i])
        except ValueError:
            with open("error.txt", "w") as f:
                f.write(str(multilabel_labels[i]))
                f.write(f"feature {feature}\n")
                f.write(f"Hamming loss preds {multilabel_preds[i]}\n")
                f.write(f"logits {multilabel_logits[i]}\n")

        precision = precision_score(
            multilabel_labels[i], multilabel_preds[i], average="macro", zero_division=0
        )
        recall = recall_score(
            multilabel_labels[i], multilabel_preds[i], average="macro", zero_division=0
        )
        f1 = f1_score(
            multilabel_labels[i], multilabel_preds[i], average="macro", zero_division=0
        )
        metrics[f"{feature}_hamming_loss"] = hamming
        metrics[f"{feature}_macro_precision"] = precision
        metrics[f"{feature}_macro_recall"] = recall
        metrics[f"{feature}_macro_f1"] = f1

    avg_macro_f1 = 0
    for feature in data.MULTICLASS_FEATURES + data.MULTILABEL_FEATURES:
        avg_macro_f1 += metrics[f"{feature}_macro_f1"]
    avg_macro_f1 /= len(data.MULTICLASS_FEATURES + data.MULTILABEL_FEATURES)
    metrics["avg_macro_f1"] = avg_macro_f1

    return metrics
