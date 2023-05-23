import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.nn.functional as nnf
from transformers import BertModel


class NeuralImageCaptioner(nn.Module):
    def __init__(self, image_encoder, vocab_size):
        super(NeuralImageCaptioner, self).__init__()
        self.image_encoder = image_encoder
        self.embedding = nn.Embedding(vocab_size, 512, padding_idx=0, max_norm=1)
        self.vocab_size = vocab_size

        # freeze image and text encoders
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.caption_decoder = nn.LSTM(
            input_size=512,
            hidden_size=768
        )
        self.time_distributed = nn.Linear(768, vocab_size)

    def forward(self, pixel_values, text_inputs):
        text_features = self.embedding(text_inputs["input_ids"])
        text_lengths = (text_inputs["attention_mask"]).sum(dim=1).cpu()
        text_features = pack_padded_sequence(text_features, text_lengths, batch_first=True, enforce_sorted=False)

        image_features = self.image_encoder(pixel_values).last_hidden_state[:, 0].unsqueeze(0)
        x, (h_t, c_t) = self.caption_decoder(text_features, (image_features, torch.zeros_like(image_features)))
        x = pad_packed_sequence(x, batch_first=True, total_length=50)[0]
        logits = self.time_distributed(x)
        return logits
    
    @torch.no_grad()
    def generate(self, pixel_values, max_length):
        image_features = self.image_encoder(pixel_values).last_hidden_state[:, 0].unsqueeze(0)
        # input_ids is a zeros tensor of shape batch_size x 50, the frst token is the start token (101)
        input_ids = torch.zeros((pixel_values.shape[0], 50), dtype=torch.long, device=pixel_values.device)
        input_ids[:, 0] = 101

        for i in range(1, max_length):
            text_features = self.embedding(input_ids)
            # text_lengths is max_length repeated batch_size times
            text_lengths = torch.full((pixel_values.shape[0],), i, dtype=torch.long, device=torch.device("cpu"))
            text_features = pack_padded_sequence(text_features, text_lengths, batch_first=True, enforce_sorted=False)
            x, (h_t, c_t) = self.caption_decoder(text_features, (image_features, torch.zeros_like(image_features)))
            x = pad_packed_sequence(x, batch_first=True, total_length=max_length)[0]
            logits = self.time_distributed(x)
            input_ids[:, i] = logits[:, i-1].argmax(dim=1)

        # if last index is not 102, put 102 at the end
        for i in range(input_ids.shape[0]):
            if input_ids[i, -1] != 102:
                input_ids[i, -1] = 102

        # at each row of input_ids, put 0s after the first 102 token
        for i in range(input_ids.shape[0]):
            input_ids[i, input_ids[i, :].tolist().index(102)+1:] = 0

        return input_ids

