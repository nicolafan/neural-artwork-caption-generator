import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from evaluate import load


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

augment_image = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1)),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomRotation(degrees=10),
])


def _transform_train(example_batch, tokenizer, image_processor, max_length):
    images = [augment_image(x) for x in example_batch["image"]]
    inputs = tokenizer(
        example_batch["caption"], 
        max_length=max_length, 
        truncation=True, 
        padding="max_length",
        return_tensors="pt",
    )

    pixel_values = image_processor(images, return_tensors="pt").pixel_values
    inputs.update({
        "pixel_values": pixel_values,
        "clip_score": torch.tensor(example_batch["clip_score"]),
        "file_name": example_batch["file_name"]
    })
    return inputs
    

def _transform_test(example_batch, tokenizer, image_processor, max_length):
    # just compute the input_ids for all captions in example
    images = [x for x in example_batch["image"]]
    captions_list = [x for x in example_batch["captions"]]
    clip_scores = torch.tensor([x for x in example_batch["clip_score"]])
    file_names = [x for x in example_batch["file_name"]]
    inputs_list = [
        tokenizer(
            captions,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        for captions in captions_list
    ]
    # from list of dicts to dict of lists
    inputs = {}
    for k in inputs_list[0].keys():
        inputs[k] = [x[k] for x in inputs_list]

    pixel_values = image_processor(images, return_tensors="pt").pixel_values
    inputs.update({
        "pixel_values": pixel_values,
        "clip_score": clip_scores,
        "file_name": file_names
    })
    return inputs


@torch.no_grad()
def compute_metrics(model, processor, test_dataloader, max_length):
    bleu = load("bleu")
    meteor = load("meteor")
    rouge = load("rouge")

    all_decoded_predictions, all_decoded_labels = [], []
    for batch in tqdm(test_dataloader):
        labels = batch.pop("input_ids").to(DEVICE)
        pixel_values = batch.pop("pixel_values").to(DEVICE)

        generated_ids = model.generate(pixel_values=pixel_values, max_length=max_length)
        decoded_labels = []
        for label_group in labels:
            decoded_labels.append(processor.batch_decode(label_group, skip_special_tokens=True))
        decoded_predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)

        all_decoded_predictions += decoded_predictions
        all_decoded_labels += decoded_labels
    
    bleu1_score = bleu.compute(predictions=all_decoded_predictions, references=all_decoded_labels, max_order=1)
    bleu4_score = bleu.compute(predictions=all_decoded_predictions, references=all_decoded_labels)
    rouge_score = rouge.compute(predictions=all_decoded_predictions, references=all_decoded_labels)
    meteor_score = meteor.compute(predictions=all_decoded_predictions, references=all_decoded_labels)
    return {
        "bleu1_score": bleu1_score,
        "bleu4_score": bleu4_score,
        "rouge_score": rouge_score,
        "meteor_score": meteor_score
    }