from transformers import ViTModel, ViTConfig
import torch.nn as nn

# Load the pre-trained ViT model
model = ViTModel.from_pretrained('google/vit-base-patch16-224')

