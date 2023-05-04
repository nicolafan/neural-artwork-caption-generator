import logging
import math
import os
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
import random

import click
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_from_disk
from dotenv import find_dotenv, load_dotenv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import ViTImageProcessor

from src.models.multiclassification import data
from src.models.multiclassification.losses import losses_fn, join_losses
from src.models.multiclassification.evaluate_model import evaluate
from src.models.multiclassification.model import ViTForMultiClassification
from src.utils.dirutils import get_data_dir
from src.utils.logutils import init_log

# set dictionaries with features and number of classes
MULTICLASS_CLASSIFICATIONS, MULTILABEL_CLASSIFICATIONS = data.get_multiclassification_dicts()
MULTICLASS_FEATURES, MULTILABEL_FEATURES = list(MULTICLASS_CLASSIFICATIONS.keys()), list(MULTILABEL_CLASSIFICATIONS.keys())
ALL_FEATURES = MULTICLASS_FEATURES + MULTILABEL_FEATURES
# set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def remove_useless_features(feature):
    global MULTICLASS_CLASSIFICATIONS, MULTILABEL_CLASSIFICATIONS
    if feature in MULTICLASS_CLASSIFICATIONS:
        MULTICLASS_CLASSIFICATIONS = {feature: MULTICLASS_CLASSIFICATIONS[feature]}
        MULTILABEL_CLASSIFICATIONS = {}
    elif feature in MULTILABEL_CLASSIFICATIONS:
        MULTILABEL_CLASSIFICATIONS = {feature: MULTILABEL_CLASSIFICATIONS[feature]}
        MULTICLASS_CLASSIFICATIONS = {}
    
    global MULTICLASS_FEATURES, MULTILABEL_FEATURES, ALL_FEATURES
    MULTICLASS_FEATURES, MULTILABEL_FEATURES = list(MULTICLASS_CLASSIFICATIONS.keys()), list(MULTILABEL_CLASSIFICATIONS.keys())
    ALL_FEATURES = MULTICLASS_FEATURES + MULTILABEL_FEATURES


def find_last_checkpoint(dir):
    # find last .pt file split on - and take the last part
    return sorted(dir.glob("*.pt"), key=lambda x: int(x.stem.split("-")[-1].replace(".pt", "")))[-1]


def train_one_epoch(model, optimizer, dataloader, batch_size, num_accumulation_steps):
    epoch_loss = 0.
    epoch_label_losses = [0.] * len(ALL_FEATURES)
    n_batches = math.ceil(len(dataloader) * batch_size / 32) # virtual number of batches

    optimizer.zero_grad()
    for i, batch in enumerate(tqdm(dataloader)):
        # get inputs and targets from batch
        inputs, targets = batch["pixel_values"], {k: batch[k] for k in batch if k != "pixel_values"}

        # predict
        outputs = model(inputs)

        # Compute the loss and its gradients
        losses = losses_fn(MULTICLASS_FEATURES, MULTILABEL_FEATURES, outputs, targets)
        loss = join_losses(model, losses)
        loss = loss / num_accumulation_steps
        loss.backward()
        
        # Update our running losses and loss tally
        for j, _ in enumerate(ALL_FEATURES):
            epoch_label_losses[j] += losses[j].item() / num_accumulation_steps
        epoch_loss += loss.item()

        if (i+1) % num_accumulation_steps == 0 or (i+1) == len(dataloader):
            # Update weights
            optimizer.step()
            optimizer.zero_grad()

    # return the average loss and the average loss per label
    return epoch_loss / n_batches, [loss / n_batches for loss in epoch_label_losses]


@click.command()
@click.option(
    "--model-output-dir",
    type=click.Path(exists=False, file_okay=False),
    help="Directory where the model will be saved.",
)
@click.option(
    "--feature",
    type=click.Choice(data.MULTICLASS_FEATURES + data.MULTILABEL_FEATURES),
    default=None,
    help="Feature to train the model for, if None use multitask learning.",
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
@click.option(
    "--resume-from-checkpoint/--no-resume-from-checkpoint",
    default=False,
    help="Whether to resume from checkpoint.",
)
def train(model_output_dir, feature, freeze_base_model, epochs, batch_size, learning_rate, resume_from_checkpoint):
    logger = logging.getLogger(__name__)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model_output_dir = Path(model_output_dir)
    
    # load and process dataset
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    dataset: Dataset = load_from_disk(get_data_dir() / "processed" / "multiclassification_dataset")
    class_weight_tensors = data.compute_class_weight_tensors(dataset, DEVICE)

    dataset = dataset.with_transform(partial(data.transform_for_model, processor=processor, device=DEVICE))
    train_loader = torch.utils.data.DataLoader(dataset["train"], batch_size=batch_size)
    validation_loader = torch.utils.data.DataLoader(dataset["validation"], batch_size=batch_size)

    if feature is not None:
        remove_useless_features(feature)

    # load model
    model = ViTForMultiClassification(MULTICLASS_CLASSIFICATIONS, MULTILABEL_CLASSIFICATIONS, class_weight_tensors)
    model = model.to(DEVICE)
    model.freeze_base_model(freeze_base_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    epoch = 1
    num_accumulation_steps = 32 // batch_size

    # load checkpoint if needed
    if resume_from_checkpoint:
        last_checkpoint = find_last_checkpoint(model_output_dir)

        logger.info(f"loading checkpoint from {last_checkpoint}")
        checkpoint = torch.load(last_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"optimizer: {checkpoint['optimizer_state_dict']['param_groups'][0]['lr']}")
        # Compare the loaded parameters to the saved parameters
        for name, param in model.named_parameters():
            if name in checkpoint["model_state_dict"]:
                if not torch.all(torch.eq(param, checkpoint["model_state_dict"][name])):
                    print(f"Parameter {name} is not equal to saved checkpoint")
            else:
                print(f"Parameter {name} not found in saved checkpoint")
        epoch = checkpoint["epoch"]
        epoch += 1

    logger.info(f"optimizer: {optimizer.param_groups[0]['lr']}")
    # Initializing
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(model_output_dir / f"runs/{timestamp}")

    while epoch <= epochs:
        logger.info(f"EPOCH {epoch}:")

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        train_loss, train_label_losses = train_one_epoch(model, optimizer, train_loader, batch_size, num_accumulation_steps)
        torch.cuda.empty_cache()

        # Log train losses
        for i, label in enumerate(ALL_FEATURES):
            writer.add_scalar(f"train/{label}_loss", train_label_losses[i], epoch)
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
        if model.log_vars is not None:
            for i, label in enumerate(ALL_FEATURES):
                writer.add_scalar(f"train/{label}_log_var", model.log_vars[i].item(), epoch)

        # We don't need gradients on to do reporting
        model.train(False)
        avg_vloss, running_label_vlosses, metrics = evaluate(model, validation_loader, MULTICLASS_FEATURES, MULTILABEL_FEATURES) 
        logger.info('LOSS train {} valid {}'.format(train_loss, avg_vloss))

        # Log valid losses
        for i, label in enumerate(ALL_FEATURES):
            writer.add_scalar(f"valid/{label}_loss", running_label_vlosses[i] / len(validation_loader), epoch)
        writer.add_scalar("valid/loss", avg_vloss, epoch)

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('train_vs_valid_loss',
                        {'train': train_loss, 'valid': avg_vloss},
                        epoch)
        
        # Log metrics
        for k, v in metrics.items():
            writer.add_scalar(f'valid/{k}', v, epoch)
        writer.flush()

        # Save the model at each epoch
        model_path = model_output_dir / f"model-{timestamp}-{epoch}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), 
            "epoch": epoch}, model_path)

        epoch += 1


if __name__ == "__main__":
    init_log()
    load_dotenv(find_dotenv())

    train()