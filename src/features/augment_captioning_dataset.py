from functools import partial

import nlpaug.augmenter.word as naw
from datasets import load_from_disk

from src.utils.dirutils import get_data_dir


def main():
    data_dir = get_data_dir() / "processed"
    dataset = load_from_disk(data_dir / "captioning_dataset")

    back_translation_aug_fr = naw.BackTranslationAug(
        from_model_name="Helsinki-NLP/opus-mt-en-fr",
        to_model_name="Helsinki-NLP/opus-mt-fr-en",
        device="cuda",
        batch_size=16,
    )

    back_translation_aug_jap = naw.BackTranslationAug(
        from_model_name="Helsinki-NLP/opus-mt-en-jap",
        to_model_name="Helsinki-NLP/opus-mt-jap-en",
        device="cuda",
        batch_size=16,
    )

    def _captions_as_lists(examples):
        captions_as_lists = [[caption] for caption in examples["caption"]]
        examples["captions"] = captions_as_lists
        return examples

    dataset = dataset.map(_captions_as_lists, batched=True, remove_columns=["caption"])

    def _augment_captions(examples, augmenter):
        aug_captions = augmenter.augment(
            [
                captions[0].replace("The artwork depicts ", "")
                for captions in examples["captions"]
            ]
        )
        for i, captions in enumerate(examples["captions"]):
            captions.append("The artwork depicts " + aug_captions[i])
        return examples
    dataset = dataset.map(partial(_augment_captions, augmenter=back_translation_aug_fr), batched=True)
    dataset = dataset.map(partial(_augment_captions, augmenter=back_translation_aug_jap), batched=True)
    dataset.save_to_disk(data_dir / "processed" / "augmented_captioning_dataset")
    

if __name__ == "__main__":
    main()
