from src.models.multiclassification.model import ViTForMultiClassification
from src.models.multiclassification.data import get_multiclassification_dicts
from src.utils.dirutils import get_data_dir, get_models_dir
from joblib import load
import numpy as np
from transformers import ViTImageProcessor
import torch
from functools import partial


class ViTForMultiClassificationPredictor:
    def __init__(self, model_path, device, batch_size=1):
        self.multiclassification_dicts = get_multiclassification_dicts()
        self.model = self.load_model(model_path, device)

        self.ordinal_encoders = {}
        self.multilabel_binarizers = {}
        for feature in self.multiclassification_dicts[0].keys():
            ordinal_encoder = load(get_data_dir() / "processed" / "ordinal_encoders" / f"{feature}.joblib")
            self.ordinal_encoders[feature] = ordinal_encoder
        for feature in self.multiclassification_dicts[1].keys():
            multilabel_binarizer = load(get_data_dir() / "processed" / "multilabel_binarizers" / f"{feature}.joblib")
            self.multilabel_binarizers[feature] = multilabel_binarizer
        
        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

        self.device = device
        self.batch_size = batch_size

    def load_model(self, model_path, device):
        model = ViTForMultiClassification(*self.multiclassification_dicts)
        checkpoint = torch.load(get_models_dir() / model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.train(False)
        return model
    
    @torch.no_grad()
    def predict(self, image):
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)
        outputs = self.model(pixel_values)

        predicted_classes = {}
        predicted_labels = {}
        for feature, output in outputs.items():
            if feature in self.multiclassification_dicts[0]:
                predicted_class = torch.argmax(output, dim=1).item()
                predicted_class = self.ordinal_encoders[feature].inverse_transform([[predicted_class]])[0]
                predicted_classes[feature] = predicted_class
            elif feature in self.multiclassification_dicts[1]:
                predicted_labels_example = torch.where(output > 0, 1, 0).cpu().numpy()
                predicted_labels_example = self.multilabel_binarizers[feature].inverse_transform(predicted_labels_example)[0]
                predicted_labels[feature] = predicted_labels_example
        
        return predicted_classes, predicted_labels


