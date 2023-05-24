import json
import os
import random
from functools import partial

import numpy as np
import torch
import torch.nn.functional as nnf
from datasets import load_from_disk
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
from transformers import (
    BertTokenizerFast,
    ViTConfig,
    ViTImageProcessor,
    ViTModel,
    get_cosine_schedule_with_warmup,
)

from src.models.captioning.nic.model import NeuralImageCaptioner
from src.models.captioning.utils import (
    _transform_test,
    _transform_train,
    compute_metrics,
    get_multicassification_vit_tools,
)
from src.utils.dirutils import get_data_dir, get_models_dir

MAX_LENGTH = 50
IDEAL_BATCH_SIZE = 64
EPOCHS = 5
OUTPUT_DIR = get_models_dir() / "captioning" / "nic"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = IDEAL_BATCH_SIZE // BATCH_SIZE

# set all seeds
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    dataset = load_from_disk(
        get_data_dir() / "processed" / "captioning_dataset_augmented_processed"
    )
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    image_processor, image_encoder = get_multicassification_vit_tools()

    dataset["train"].set_transform(
        partial(
            _transform_train,
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_length=MAX_LENGTH,
        )
    )
    dataset["validation"].set_transform(
        partial(
            _transform_test,
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_length=MAX_LENGTH,
        )
    )
    dataset["test"].set_transform(
        partial(
            _transform_test,
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_length=MAX_LENGTH,
        )
    )
    train_dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=BATCH_SIZE)
    validation_dataloader = torch.utils.data.DataLoader(
        dataset["validation"], batch_size=BATCH_SIZE
    )

    model = NeuralImageCaptioner(image_encoder, len(tokenizer))
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(dataset["train"]) // IDEAL_BATCH_SIZE) * EPOCHS,
    )
    running_loss = 0.0
    epoch = 0

    checkpoint = torch.load(get_models_dir() / "captioning" / "nic" / "0.pt", map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint["epoch"] + 1

    while epoch < EPOCHS:
        model.train(True)
        for step, batch in enumerate(tqdm(train_dataloader), start=1):
            text_inputs = {
                "input_ids": batch.pop("input_ids").to(DEVICE),
                "attention_mask": batch.pop("attention_mask").to(DEVICE),
                "token_type_ids": batch.pop("token_type_ids").to(DEVICE),
            }
            pixel_values, clip_weights = (
                batch.pop("pixel_values").to(DEVICE),
                batch.pop("clip_score").to(DEVICE) * 8 - 1 / 5,
            )

            logits = model(pixel_values, text_inputs)
            shifted_logits = logits[:, :-1, :].contiguous()
            labels = text_inputs["input_ids"][:, 1:].contiguous()
            shifted_logits = shifted_logits.permute(0, 2, 1)
            losses = nnf.cross_entropy(
                shifted_logits,
                labels,
                ignore_index=tokenizer.pad_token_id,
                reduction="none",
            )
            losses = losses.sum(dim=1) / (labels != tokenizer.pad_token_id).sum(dim=1)
            weighted_losses = losses * clip_weights
            loss = weighted_losses.mean() / GRADIENT_ACCUMULATION_STEPS
            running_loss += loss.item()

            loss.backward()

            if step % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        model.train(False)
        avg_loss = running_loss / (len(dataset["train"]) // IDEAL_BATCH_SIZE)
        print("Loss: ", avg_loss)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_loss,
            },
            OUTPUT_DIR / f"{epoch}.pt",
        )
        metrics = compute_metrics(model, tokenizer, validation_dataloader, MAX_LENGTH)
        metrics["loss"] = avg_loss
        with open(os.path.join(OUTPUT_DIR, f"{epoch}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        epoch += 1


if __name__ == "__main__":
    main()


