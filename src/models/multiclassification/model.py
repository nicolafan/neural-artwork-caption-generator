import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel


class ViTForMultiClassification(nn.Module):
    def __init__(
        self,
        multiclass_classifications: dict[str, int],
        multilabel_classifications: dict[str, int],
        multiclass_class_weights: dict[str, torch.Tensor]=None,
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

        # loss weights (if multitask learning)
        if self.n_classifications > 1:
            self.log_vars = nn.Parameter(
                torch.zeros(
                    self.n_classifications, dtype=torch.float32, requires_grad=True
                )
            )
        else:
            self.log_vars = None

    def freeze_base_model(self, freeze: bool):
        """Toggle freeze/unfreeze of the ViT model.

        Args:
            freeze (bool): freeze or unfreeze
        """
        for param in self.vit.parameters():
            param.requires_grad = not freeze

    def forward(
        self, pixel_values
    ):
        """Forward pass for ViTForMultiClassification model.

        Args:
            pixel_values (torch.Tensor): pixel values of the images
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

        logits_dict = {
            feature: logits[i]
            for i, feature in enumerate(
                list(self.multiclass_classifications.keys())
                + list(self.multilabel_classifications.keys())
            )
        }
        return logits_dict
