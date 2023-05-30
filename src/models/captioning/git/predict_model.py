import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from src.utils.dirutils import get_data_dir, get_models_dir


def get_git_generation_tools(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "microsoft/git-base"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.train(False)
    return processor, model, device


def git_generate_coco_output(model_path, dataset):
    processor, model, device = get_git_generation_tools(model_path)
    def _transform(examples):
        images = [image for image in examples["image"]]
        return processor(images=images, return_tensors="pt")
    dataset.set_transform(_transform)
    dataloader = DataLoader(dataset, batch_size=8)

    outputs = []
    for i, examples in enumerate(tqdm(dataloader)):
        pixel_values = examples["pixel_values"].to(device, torch.float16)
        generated_ids = model.generate(
            pixel_values=pixel_values,
            max_length=50,
            num_beams=1,
            no_repeat_ngram_size=2,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

        outputs.append({
            "image_id": i,
            "caption": generated_text[0]
        })
    return outputs
