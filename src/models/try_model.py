from pathlib import Path
import os
import pandas as pd
from PIL import Image
from transformers import AutoTokenizer, Blip2Processor, Blip2ForConditionalGeneration
import requests
import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"

checkpoint = "Salesforce/blip2-opt-2.7b"
processor = Blip2Processor.from_pretrained(checkpoint)
model = Blip2ForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.float16)
model.to("cuda")