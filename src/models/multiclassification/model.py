import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel


# custom loss for multilabel classification, ignore elements that have 0 labels
def binary_cross_entropy_with_logits_ignore_no_labels(preds, targets):
    zero_labels = torch.sum(targets, dim=1) == 0
    targets = targets.float() # remove this in the future

    # Apply BCEWithLogitsLoss only to non-zero labels
    loss = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
    loss = torch.where(zero_labels.unsqueeze(1), torch.zeros_like(loss), loss)
    loss = torch.mean(loss)
    
    return loss


class ViTForMultiClassification(nn.Module):
    def __init__(self, multiclass_classifications: dict[str, int], multilabel_classifications: dict[str, int], multiclass_class_weights: dict[str, torch.Tensor]):
        """Initialize a ViTForMultiClassification model for multi-classification and multi-label classification.

        Args:
            multiclass_classifications (dict[str, int]): dictionary of multiclass classification with feature names and number of classes
            multilabel_classifications (dict[str, int]): dictionary of multilabel classification with feature names and number of classes
        """
        super(ViTForMultiClassification, self).__init__()
        self.multiclass_classifications = multiclass_classifications
        self.multilabel_classifications = multilabel_classifications
        self.multiclass_class_weights = multiclass_class_weights

        # initialize ViT model
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")

        # initialize classification heads
        self.multiclass_fcs = nn.ModuleList([
            nn.Linear(self.vit.config.hidden_size, num_classes)
            for num_classes
            in multiclass_classifications.values()
        ])
        self.multilabel_fcs = nn.ModuleList([
            nn.Linear(self.vit.config.hidden_size, num_classes)
            for num_classes
            in multilabel_classifications.values()
        ])

        # initialize losses as a list with 5 0s
        self.losses = [0] * 5
        # loss weights
        self.loss_w = nn.Parameter(
            torch.ones(5, dtype=torch.float32, requires_grad=True)
        )

    def compute_losses(self, logits: tuple[torch.Tensor], artist: torch.Tensor, style: torch.Tensor, genre: torch.Tensor, tags: torch.Tensor, media: torch.Tensor):
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
            feature = list(self.multiclass_classifications.keys())[i]
            loss = F.cross_entropy(logits[i], labels.squeeze(), weight=self.multiclass_class_weights[feature], ignore_index=-1)
            if torch.isnan(loss):
                loss = 0
            losses.append(loss)

        for i, labels in enumerate([tags, media]):
            loss = binary_cross_entropy_with_logits_ignore_no_labels(logits[i + 3], labels)
            if torch.isnan(loss):
                loss = 0
            losses.append(loss)

        # sum the list of losses to self.losses
        for i, loss in enumerate(losses):
            self.losses[i] += loss if isinstance(loss, int) else loss.item()

        return losses
    
    def compute_loss(self, logits: tuple[torch.Tensor], artist: torch.Tensor, style: torch.Tensor, genre: torch.Tensor, tags: torch.Tensor, media: torch.Tensor):
        """Compute weighted loss.

        Args:
            losses (torch.Tensor): list of losses

        Returns:
            torch.Tensor: weighted loss
        """
        losses = self.compute_losses(logits, artist, style, genre, tags, media)
        eps = 1e-8
        weighted_losses = []
        loss = 0

        for i, loss in enumerate(losses):
            weighted_loss = loss / (self.loss_w[i]**2 + eps)
            weighted_losses.append(weighted_loss)

        # set regularizer to sum of log of loss weights plus eps
        regularizer = torch.sum(torch.log(self.loss_w + eps))

        # sum the list of weighted losses
        loss = torch.sum(torch.stack(weighted_losses)) + regularizer
        return loss
        

    def forward(self, pixel_values, artist=None, style=None, genre=None, tags=None, media=None):
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

        multiclass_logits = tuple(
            fc(x)
            for fc in self.multiclass_fcs
        )
        multilabel_logits = tuple(
            fc(x)
            for fc in self.multilabel_fcs
        )

        logits = multiclass_logits + multilabel_logits
        
        # compute loss if labels are provided
        if artist is not None:
            loss = self.compute_loss(logits, artist, style, genre, tags, media)
            return (loss,) + logits
        else:
            return logits
