import torch
from transformers import Trainer, TrainerCallback, TrainingArguments, EarlyStoppingCallback
from transformers.integrations import TensorBoardCallback

import src.models.multiclassification.data as data
from src.models.multiclassification.metrics import compute_metrics
from src.models.multiclassification.model import ViTForMultiClassification
from src.utils.dirutils import get_models_dir

MODEL_OUTPUT_DIR = get_models_dir() / "multiclassification" / "pretrain"


class CustomTensorBoardCallback(TensorBoardCallback):
    """Custom TensorBoard callback.
    """
    def __init__(self, *args, **kwargs):
        """Initialize CustomTensorBoardCallback.
        """
        super().__init__(*args, **kwargs)
        self.last_step = -1

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics to TensorBoard.
        """
        super().on_log(args, state, control, logs, **kwargs)
        if self.last_step == state.global_step: # skip if already logged
            return
        self.last_step = state.global_step
        model = kwargs["model"]
        # TODO: divide by the no. of steps in the epoch
        multiclass_losses = dict(
            (feature, model.losses_epoch_sum[i])
            for i, feature in enumerate(model.multiclass_classifications.keys())
        )
        multilabel_losses = dict(
            (feature, model.losses_epoch_sum[i + len(model.multiclass_classifications.keys())])
            for i, feature in enumerate(model.multilabel_classifications.keys())
        )

        for feature, loss in multiclass_losses.items():
            self.tb_writer.add_scalar(f"train/{feature}_loss", loss, state.global_step)
        for feature, loss in multilabel_losses.items():
            self.tb_writer.add_scalar(f"train/{feature}_loss", loss, state.global_step)

        log_vars = dict(
            (feature, model.log_vars[i])
            for i, feature in enumerate(
                list(model.multiclass_classifications.keys()) + list(model.multilabel_classifications.keys())
            )
        )
        for feature, log_var in log_vars.items():
            self.tb_writer.add_scalar(
                f"train/{feature}_sigma", log_var, state.global_step
            )
        self.tb_writer.flush()


class ResetLossesCallback(TrainerCallback):
    """Reset losses callback.
    """
    def on_epoch_start(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.losses_epoch_sum = [0] * 5


def train():
    """Train model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = data.get_dataset_for_multiclassification()

    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        label_names=list(data.MULTICLASS_FEATURES + data.MULTILABEL_FEATURES),
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        seed=42,
        load_best_model_at_end=True,
        metric_for_best_model="avg_macro_f1",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
    )

    model = ViTForMultiClassification(
        *data.get_multiclassification_dicts(), data.compute_class_weight_tensors(dataset, device)
    )
    model.to(device)
    model.freeze_base_model(True)

    dataset.set_format(
        type="torch",
        columns=["pixel_values", "artist", "style", "genre", "tags", "media"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        callbacks=[CustomTensorBoardCallback(), ResetLossesCallback(), EarlyStoppingCallback()],
    )
    trainer.train()


if __name__ == "__main__":
    train()