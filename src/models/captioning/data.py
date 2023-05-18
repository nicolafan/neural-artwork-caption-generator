from src.utils.dirutils import get_data_dir


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

    dataset["train"] = dataset["train"].map(_extend_examples, batched=True, num_proc=4, remove_columns=["captions"])
    dataset = dataset["train"].shuffle(seed=42)
    dataset = dataset["train"].flatten_indices()
    dataset.save_to_disk(get_data_dir() / "processed" / "captioning_dataset_augmented_prepared")