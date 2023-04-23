import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel


# custom loss for multilabel classification, ignore elements that have 0 labels
def binary_cross_entropy_with_logits_ignore_no_labels(preds, targets):
    zero_labels = torch.sum(targets, dim=-1) == 0
    targets = targets.float()

    # Apply BCEWithLogitsLoss only to non-zero labels
    loss = F.binary_cross_entropy_with_logits(preds, targets, reduction="none")
    loss = torch.where(zero_labels.unsqueeze(-1), torch.zeros_like(loss), loss)
    loss = torch.mean(loss)

    return loss


class ViTForMultiClassification(nn.Module):
    def __init__(
        self,
        multiclass_classifications: dict[str, int],
        multilabel_classifications: dict[str, int],
        multiclass_class_weights: dict[str, torch.Tensor],
    ):
        """Initialize a ViTForMultiClassification model for multi-classification and multi-label classification.

        Args:
            multiclass_classifications (dict[str, int]): dictionary of multiclass classification with feature names and number of classes
            multilabel_classifications (dict[str, int]): dictionary of multilabel classification with feature names and number of classes
            multiclass_class_weights (dict[str, torch.Tensor]): dictionary of weights for each class in each multiclass classification
        """
        super(ViTForMultiClassification, self).__init__()
        self.multiclass_classifications = multiclass_classifications
        self.multilabel_classifications = multilabel_classifications
        self.n_classifications = len(multiclass_classifications) + len(
            multilabel_classifications
        )
        self.multiclass_class_weights = multiclass_class_weights

        # initialize ViT model
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")

        # initialize classification heads
        if self.multiclass_classifications:
            self.multiclass_fcs = nn.ModuleList(
                [
                    nn.Linear(self.vit.config.hidden_size, num_classes)
                    for num_classes in multiclass_classifications.values()
                ]
            )

        if self.multilabel_classifications:
            self.multilabel_fcs = nn.ModuleList(
                [
                    nn.Linear(self.vit.config.hidden_size, num_classes)
                    for num_classes in multilabel_classifications.values()
                ]
            )

        # initialize losses as a list with 5 0s
        self.losses_epoch_sum = [0] * self.n_classifications
        # loss weights
        if self.n_classifications > 1:
            self.log_vars = nn.Parameter(
                torch.zeros(
                    self.n_classifications, dtype=torch.float32, requires_grad=True
                )
            )
        else:
            self.log_vars = None

    def toggle_freeze(self, freeze: bool):
        """Toggle freeze/unfreeze of the ViT model.

        Args:
            freeze (bool): freeze or unfreeze
        """
        for param in self.vit.parameters():
            param.requires_grad = not freeze

    def compute_losses(
        self,
        logits: tuple[torch.Tensor],
        artist: torch.Tensor,
        style: torch.Tensor,
        genre: torch.Tensor,
        tags: torch.Tensor,
        media: torch.Tensor,
    ):
        """Compute loss for multi-classification and multi-label classification.

        Args:
            logits (tuple[torch.Tensor]): output of the classification heads
            artist (torch.Tensor): artist labels
            style (torch.Tensor): style labels
            genre (torch.Tensor): genre labels
            tags (torch.Tensor): tags labels
            media (torch.Tensor): media labels

        Returns:
            torch.Tensor: loss
        """
        losses = []

        for i, labels in enumerate([artist, style, genre]):
            if labels is None:
                continue
            feature = list(self.multiclass_classifications.keys())[i]
            loss = F.cross_entropy(
                logits[i],
                labels.squeeze(),
                weight=self.multiclass_class_weights[feature],
                ignore_index=-1,
            )
            if torch.isnan(loss):
                loss = 0
            losses.append(loss)

        for i, labels in enumerate([tags, media]):
            if labels is None:
                continue
            loss = binary_cross_entropy_with_logits_ignore_no_labels(
                logits[i + len(self.multiclass_classifications)], labels
            )
            if torch.isnan(loss):
                loss = 0
            losses.append(loss)

        # sum the list of losses to self.losses
        for i, loss in enumerate(losses):
            self.losses_epoch_sum[i] += loss if isinstance(loss, int) else loss.item()

        return losses

    def compute_loss(
        self,
        logits: tuple[torch.Tensor],
        artist: torch.Tensor,
        style: torch.Tensor,
        genre: torch.Tensor,
        tags: torch.Tensor,
        media: torch.Tensor,
    ):
        """Compute weighted loss.

        Args:
            logits (tuple[torch.Tensor]): tuple of logits
            artist (torch.Tensor): artist tensor
            style (torch.Tensor): style tensor
            genre (torch.Tensor): genre tensor
            tags (torch.Tensor): tags tensor
            media (torch.Tensor): media tensor

        Returns:
            torch.Tensor: weighted loss
        """
        losses = self.compute_losses(logits, artist, style, genre, tags, media)

        # return the first loss if there is only one loss
        if len(losses) == 1:
            return losses[0]

        # Compute weighted losses
        weighted_losses = [
            loss / (torch.exp(-self.log_vars[i])) for i, loss in enumerate(losses)
        ]

        # Compute regularizer as sum of log of loss weights plus eps
        regularizer = torch.sum(self.log_vars / 2)

        # Sum the list of weighted losses
        loss = torch.sum(torch.stack(weighted_losses)) + regularizer

        return loss

    def forward(
        self, pixel_values, artist=None, style=None, genre=None, tags=None, media=None
    ):
        """Forward pass for ViTForMultiClassification model.

        Args:
            pixel_values (torch.Tensor): pixel values of the images
            artist (torch.Tensor, optional): artist labels. Defaults to None.
            style (torch.Tensor, optional): style labels. Defaults to None.
            genre (torch.Tensor, optional): genre labels. Defaults to None.
            tags (torch.Tensor, optional): tags labels. Defaults to None.
            media (torch.Tensor, optional): media labels. Defaults to None.
        """
        x = self.vit(pixel_values=pixel_values).pooler_output
        logits = None

        if self.multiclass_classifications:
            multiclass_logits = tuple(fc(x) for fc in self.multiclass_fcs)
            logits = multiclass_logits
        if self.multilabel_classifications:
            multilabel_logits = tuple(fc(x) for fc in self.multilabel_fcs)
            if logits is not None:
                logits = logits + multilabel_logits
            else:
                logits = multilabel_logits

        # compute loss only if there is at least one not None label
        if (
            artist is not None
            or style is not None
            or genre is not None
            or tags is not None
            or media is not None
        ):
            loss = self.compute_loss(logits, artist, style, genre, tags, media)
            return (loss,) + logits
        else:
            return logits
