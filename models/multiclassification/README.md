# Multiclassification Model

The multiclassification model uses Vision Transformer (ViT), pre-trained on ImageNet-21k ([google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k)).

ViT is used to solve a multitask classification problem, based on the content of ArtGraph. For each artwork present on ArtGraph (116475 artworks), we take its **artist**, **style**, **genre**, **tags**, and **media**, and put a multiclass/multilabel classification head on top of ViT for each information to predict. Each artwork in ArtGraph is assigned to a style, genre, or artist. Not all artworks are assigned to tags or media (not because they are not related to one of them, but probably because of missing information on WikiArt). Predicting the artist, style, or genre of the artwork consists in performin a multiclass classification. Predicting tags or media consists in performing a multilabel classification, since multiple tags or media can be associated with an artwork.

For the classification of the artists, tags, and media, we only consider the ones that are associated to 100 or more paintings. The artist `other` has been added to the lists of artists to classify, and it has been associated to all the artworks whose artist has been removed because of little representation.

## Multitask Learning

The 5 classification problems are associated to 5 different cross-entropy losses. It is clear that the losses are strictly related on to another (which is also the reason why we hope to obtain better results on the classification task if we use multitask learning, to learn common features), and they could present different magnitudes.

For this reason, instead of associating a weight to each loss, we let the model learn the weights of the losses, using the strategy discussed in the paper ["Multi-task learning using uncertainty to weigh losses for scene geometry and semantics"](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf).

Our loss is presented as: $$L = \sum_{i=1}^{N_{loss}}\frac{L_i}{\sigma_i^2+\epsilon} + \sum_{i=1}^{N_{loss}}log(\sigma_i+\epsilon)$$
We learn 5 log variances, one for each task, and apply them to the computation of the loss, also letting the model change the log variances with backpropagation.

We also weighted the multiclass multiclassification losses on the different classes with class weights, and didn't consider in the computation of the multilabel losses the examples that had no media or tag associated.

## Dataset

The dataset is composed of the 116475 images of the ArtGraph, divided according to the split that can be found in this project.

## Training

The model has been trained for 35 epochs. For the first 5 epochs, we froze ViT, and trained the model heads (with log variances) using the default AdamW optimizer configuration, with a learning rate of 1e-3 and a batch size of 32. Then, we unfroze the model, training for 30 epochs, with the default AdamW optimizer, a batch size of 32 and a learning rate of 5e-5. We also added norm gradient clipping to a max norm of 1 and a dropoout layer, just before the heads of the model, with a probability of 0.3.

## Training Results

### Task Losses

![image](/reports/figures/multiclassification_losses.png)

### Task Macro F1 Scores (Validation)

![image](/reports/figures/multiclassification_macro_f1s.png)

## Test Results

We choose the model with the highest average macro F1 score, across al the tasks, which also corresponds to the model obtained after performing the 35th epoch of the training process. This could also mean that continuing training could have improved the classification results.

We present the following test results:

```
(92.65286856744216,
 [2.35425907741464,
  2.991217190909951,
  2.865208080331326,
  0.021898770528032057,
  0.06884663572904208],
 {'artist_accuracy': 0.6993105786618445,
  'artist_macro_precision': 0.6154203665051624,
  'artist_macro_recall': 0.5829196050233618,
  'artist_macro_f1': 0.5862963720648307,
  'style_accuracy': 0.5997965641952984,
  'style_macro_precision': 0.5841171050661121,
  'style_macro_recall': 0.5728221705935834,
  'style_macro_f1': 0.5740512721960334,
  'genre_accuracy': 0.7277915913200723,
  'genre_macro_precision': 0.678110908004751,
  'genre_macro_recall': 0.6468839532675804,
  'genre_macro_f1': 0.6594409792772451,
  'tags_hamming_loss': 0.005687743253142625,
  'tags_macro_precision': 0.5864382137369426,
  'tags_macro_recall': 0.3273901734170846,
  'tags_macro_f1': 0.3961297113144862,
  'media_hamming_loss': 0.024620611658709015,
  'media_macro_precision': 0.71293593738081,
  'media_macro_recall': 0.3719696826873494,
  'media_macro_f1': 0.46162278125964595,
  'avg_macro_f1': 0.5355082232224483})
```

The first number corresponds to the total loss, computed also considering the apprehended log variances. This number showed not to be very meaningful aside from the training process.

We show, in order, the losses for the artist, genre, style, tags, and media classifications, and, finally, the results.

