import click
import torch
from datasets import Dataset, load_from_disk
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    EarlyStoppingCallback,
    ViTImageProcessor
)
from transformers.integrations import TensorBoardCallback

import src.models.multiclassification.data as data
from src.models.multiclassification.metrics import compute_metrics
from src.models.multiclassification.model import ViTForMultiClassification
from src.utils.dirutils import get_models_dir, get_data_dir


class CustomTensorBoardCallback(TensorBoardCallback):
    """Custom TensorBoard callback."""

    def __init__(self, *args, **kwargs):
        """Initialize CustomTensorBoardCallback."""
        super().__init__(*args, **kwargs)
        self.last_step = -1

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics to TensorBoard."""
        super().on_log(args, state, control, logs, **kwargs)
        if self.last_step == state.global_step:  # skip if already logged
            return
        self.last_step = state.global_step
        model = kwargs["model"]
        # TODO: divide by the no. of steps in the epoch
        multiclass_losses = dict(
            (feature, model.losses_epoch_sum[i] / (state.global_step / state.epoch))
            for i, feature in enumerate(model.multiclass_classifications.keys())
        )
        multilabel_losses = dict(
            (
                feature,
                model.losses_epoch_sum[i + len(model.multiclass_classifications.keys())]
                / (state.global_step / state.epoch)
            )
            for i, feature in enumerate(model.multilabel_classifications.keys())
        )

        for feature, loss in multiclass_losses.items():
            self.tb_writer.add_scalar(f"train/{feature}_loss", loss, state.global_step)
        for feature, loss in multilabel_losses.items():
            self.tb_writer.add_scalar(f"train/{feature}_loss", loss, state.global_step)

        log_vars = dict(
            (feature, model.log_vars[i])
            for i, feature in enumerate(
                list(model.multiclass_classifications.keys())
                + list(model.multilabel_classifications.keys())
            )
        )
        for feature, log_var in log_vars.items():
            self.tb_writer.add_scalar(
                f"train/{feature}_log_var", log_var, state.global_step
            )
        self.tb_writer.flush()


class ResetLossesCallback(TrainerCallback):
    """Reset losses callback."""

    def on_epoch_start(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.losses_epoch_sum = [0] * 5


# add click options
@click.command()
@click.option(
    "--model-output-dir",
    type=click.Path(exists=False, file_okay=False),
    help="Directory to save the model to.",
)
@click.option(
    "--label",
    type=click.Choice(data.MULTICLASS_FEATURES + data.MULTILABEL_FEATURES),
    help="Label to train the model for.",
    default=None,
)
@click.option(
    "--freeze-base-model/--no-freeze-base-model",
    default=True,
    help="Whether to freeze the base model.",
)
@click.option(
    "--epochs",
    type=int,
    default=10,
    help="Number of epochs to train for.",
)
@click.option(
    "--batch-size",
    type=int,
    default=16,
    help="Batch size to use for training.",
)
@click.option(
    "--learning-rate",
    type=float,
    default=5e-5,
    help="Learning rate to use for training.",
)
def train(model_output_dir, label, freeze_base_model, epochs, batch_size, learning_rate):
    """Train model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset: Dataset = load_from_disk(
        get_data_dir() / "processed" / "multiclassification_dataset"
    )
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    multiclass_classifications, multilabel_classifications = data.get_multiclassification_dicts()
    label_names = list(multiclass_classifications.keys()) + list(multilabel_classifications.keys())
    if label is not None:
        if label in multiclass_classifications:
            multiclass_classifications = {label: multiclass_classifications[label]}
            multilabel_classifications = {}
        else:
            multiclass_classifications = {}
            multilabel_classifications = {label: multilabel_classifications[label]}
        label_names = [label]

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        label_names=label_names,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        seed=42,
        num_train_epochs=epochs,
        load_best_model_at_end=True,
        metric_for_best_model="avg_macro_f1",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=32 // batch_size,
        remove_unused_columns=False,
        learning_rate=learning_rate,
    )

    model = ViTForMultiClassification(
        multiclass_classifications, multilabel_classifications,
        data.compute_class_weight_tensors(dataset, device),
    )
    model.to(device)
    model.freeze_base_model(freeze_base_model)

    dataset = dataset.with_format(type="torch", columns=["image"] + label_names)

    def transform(examples):
        examples["pixel_values"] = processor(
            examples["image"], return_tensors="pt"
        ).pixel_values
        return {k: v for k, v in examples.items() if k != "image"}
    dataset.set_transform(transform)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        callbacks=[
            CustomTensorBoardCallback(),
            ResetLossesCallback(),
            EarlyStoppingCallback(early_stopping_patience=1),
        ],
    )
    trainer.train(resume_from_checkpoint=False)


if __name__ == "__main__":
    train()
