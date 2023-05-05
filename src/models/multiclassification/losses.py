import torch
import torch.nn.functional as F


def binary_cross_entropy_with_logits_ignore_no_labels(preds: torch.Tensor, targets: torch.Tensor):
    """Compute binary cross entropy with logits ignoring no-labels.
    
    Args:
        preds (torch.Tensor): tensor of predictions
        targets (torch.Tensor): tensor of targets
        
    Returns:
        torch.Tensor: binary cross entropy with logits ignoring no-labels
    """
    zero_labels = torch.sum(targets, dim=-1) == 0
    targets = targets.float()

    # Apply BCEWithLogitsLoss only to non-zero labels
    loss = F.binary_cross_entropy_with_logits(preds, targets, reduction="none")
    loss = torch.where(zero_labels.unsqueeze(-1), torch.zeros_like(loss), loss)
    loss = torch.mean(loss)

    return loss


def losses_fn(multiclass_features, multilabel_features, outputs, targets, class_weight_tensors):
    losses = []
    for feature in multiclass_features:
        loss = F.cross_entropy(outputs[feature], targets[feature].squeeze(), ignore_index=-1, weight=class_weight_tensors[feature])
        losses.append(loss)
    for feature in multilabel_features:
        loss = binary_cross_entropy_with_logits_ignore_no_labels(outputs[feature], targets[feature])
        losses.append(loss)

    return losses


def join_losses(model, losses):
    final_losses = []
    if model.log_vars is not None:
        for i, loss in enumerate(losses):
            final_losses.append(torch.exp(-model.log_vars[i]) * loss + model.log_vars[i]/2)
    else:
        final_losses = losses
    return sum(final_losses)