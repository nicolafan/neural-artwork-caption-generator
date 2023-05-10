import json
import re

import spacy


class SpacyTokenizer:
    def __init__(self, max_length=20, min_occurrences=5):
        self.max_length = max_length
        self.min_occurences = min_occurrences

        self.nlp = spacy.load("en_core_web_sm")
        self.vocab = {
            "<pad>": 0,
            "<start>": 1,
            "<end>": 2,
            "<unk>": 3,
        }

    def _clean(self, text):
        text = text.replace("The artwork depicts ", "")
        # remove punctuation
        text = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            text,
        )
        # remove multiple spaces
        text = re.sub(
            r"\s{2,}",
            " ",
            text,
        )
        # remove leading and trailing spaces
        text = text.rstrip("\n")
        text = text.strip(" ")
        return text

    def fit(self, texts):
        occurrences = {}
        for text in texts:
            text = self._clean(text)
            doc = self.nlp(text)

            # lower words in caption if they are not named entities
            text_tokens = [
                token.text.lower() if token.ent_type_ == "" else token.text
                for token in doc
            ]
            for token in text_tokens:
                if token not in occurrences:
                    occurrences[token] = 0
                occurrences[token] += 1
                if (
                    occurrences[token] >= self.min_occurences
                    and token not in self.vocab
                ):
                    self.vocab[token] = len(self.vocab)

    def transform(self, texts):
        sequences = []
        for text in texts:
            text = self._clean(text)
            doc = self.nlp(text)

            # lower words in caption if they are not named entities
            text_tokens = [
                token.text.lower() if token.ent_type_ == "" else token.text
                for token in doc
            ]
            sequence = [self.vocab["<start>"]]
            for token in text_tokens:
                if token in self.vocab:
                    sequence.append(self.vocab[token])
                else:
                    sequence.append(self.vocab["<unk>"])

            # cut up to max length
            if len(sequence) > self.max_length - 1:
                sequence = sequence[: self.max_length - 1]

            sequence.append(self.vocab["<end>"])
            sequences.append(sequence)
        return sequences

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    @classmethod
    def from_json(cls, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        return cls(**data)
