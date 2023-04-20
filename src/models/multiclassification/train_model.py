import torch
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.integrations import TensorBoardCallback

import src.models.multiclassification.data as data
from src.models.multiclassification.metrics import compute_metrics
from src.models.multiclassification.model import ViTForMultiClassification
from src.utils.dirutils import get_models_dir

MODEL_OUTPUT_DIR = get_models_dir() / "multiclassification" / "vit_base_patch16_224"


class CustomTensorBoardCallback(TensorBoardCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_step = -1

    def on_log(self, args, state, control, logs=None, **kwargs):
        super().on_log(args, state, control, logs, **kwargs)
        if self.last_step == state.global_step:
            return
        self.last_step = state.global_step
        model = kwargs["model"]
        multiclass_losses = dict(
            (feature, model.losses[i])
            for i, feature in enumerate(data.MULTICLASS_FEATURES)
        )
        multilabel_losses = dict(
            (feature, model.losses[i + len(data.MULTICLASS_FEATURES)])
            for i, feature in enumerate(data.MULTILABEL_FEATURES)
        )

        for feature, loss in multiclass_losses.items():
            self.tb_writer.add_scalar(f"train/{feature}_loss", loss, state.global_step)
        for feature, loss in multilabel_losses.items():
            self.tb_writer.add_scalar(f"train/{feature}_loss", loss, state.global_step)

        sigmas = dict(
            (feature, model.loss_w[i])
            for i, feature in enumerate(
                data.MULTICLASS_FEATURES + data.MULTILABEL_FEATURES
            )
        )
        for feature, sigma in sigmas.items():
            self.tb_writer.add_scalar(
                f"train/{feature}_sigma", sigma, state.global_step
            )
        self.tb_writer.flush()


class ResetLossesCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.losses = [0] * 5


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = data.get_dataset_for_multiclassification()
    dataset.set_format(
        type="torch",
        columns=["pixel_values", "artist", "style", "genre", "tags", "media"],
    )

    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        label_names=list(data.MULTICLASS_FEATURES + data.MULTILABEL_FEATURES),
        evaluation_strategy="steps",
        eval_steps=30,
        logging_strategy="steps",
        logging_steps=30,
    )

    model = ViTForMultiClassification(
        *data.get_multiclassification_dicts(), data.compute_class_weight_tensors()
    )
    model.to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"].select(range(200)),
        eval_dataset=dataset["validation"].select(range(200)),
        compute_metrics=compute_metrics,
        callbacks=[CustomTensorBoardCallback(), ResetLossesCallback()],
    )
    trainer.train()
