from src.utils.dirutils import get_data_dir
from datasets import load_from_disk
import re
import spacy


def _clean_text(text):
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
        # keep only 40 first words
        text = " ".join(text.split(" ")[:40])
        # remove leading and trailing spaces
        text = text.rstrip("\n")
        text = text.strip(" ")
        return text


def _process_caption(nlp, caption):
    caption = _clean_text(caption)
    doc = nlp(caption)
    # make all words lowercase if they are not named entities and preserve spaces
    caption = "".join(
        [
            (word.text.lower() + word.whitespace_) if not word.ent_type_ else (word.text + word.whitespace_)
            for word in doc
        ]
    )
    return caption


def extend_augmented_dataset(dataset):
    def _extend_examples(examples):
        new_examples = dict((k, []) for k in examples.keys() if k != "captions")
        new_examples["caption"] = []

        for i in range(len(examples["image"])):
            for caption in examples["captions"][i]:
                for k in new_examples.keys():
                    if k != "caption":
                        new_examples[k].append(examples[k][i])
                    else:
                        new_examples[k].append(caption)

        return new_examples

    dataset = dataset.map(_extend_examples, batched=True, num_proc=4, remove_columns=["captions"])
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.flatten_indices()
    return dataset


def clean_captions_dataset(dataset):
    nlp = spacy.load("en_core_web_sm")

    def _clean_captions_examples(examples):
        if "caption" in examples:
            examples["caption"] = [_process_caption(nlp, caption) for caption in examples["caption"]]
        elif "captions" in examples:
            examples["captions"] = [[_process_caption(nlp, caption) for caption in captions] for captions in examples["captions"]]
        return examples
    
    dataset = dataset.map(_clean_captions_examples, batched=True, num_proc=4)
    return dataset


def keep_good_examples(dataset):
    def condition(example):
        return example["clip_score"] >= 0.15
    
    dataset = dataset.filter(condition)
    return dataset


def main():
    dataset = load_from_disk(get_data_dir() / "processed" / "captioning_dataset_augmented")
    dataset["train"] = extend_augmented_dataset(dataset["train"])
    dataset = clean_captions_dataset(dataset)
    dataset = keep_good_examples(dataset)
    dataset.save_to_disk(get_data_dir() / "processed" / "captioning_dataset_augmented_processed")


if __name__ == "__main__":
    main()
    