# Microsoft GIT-base

The model was fine-tuned using the `captioning_dataset_augmented_processed` dataset, containing the augmented captions (three per image), and after the examples with a CLIPScore < 0.15 were removed from the dataset (all splits).

We weight the instances in the training set by their CLIPScore. The instance with the smallest allowed CLIPScore (0.15) is weighted 1, the instance with the ~highest CLIPScore (0.4) is weighted 3. All the other instances follow the weight scheme defined by the line passing between the two extreme points.

We fine-tune the model for 5 epochs, with a batch size of 64 (achieved with gradient accumulation). We adopt the AdamW default optimizer with a learning rate of `4.5e-7 ~= 2.5e-6 / 2 (~average CLIP weight) / sqrt(512(paper bs) / 64)`. We leverage a cosine schedule with warmup for 500 steps.

**We freeze the CLIP image encoder and the embedding layer**.

We stop training after three epochs, because of minor improvements in terms of metrics (BLEU1, BLEU4, METEOR, ROUGE) computed on the validation set, between the second and the third epoch.