# Artgraph Split

The dataset has been split using the following procedure:

1. The 500 (image, caption) examples that were checked by a human were randomly divided into two groups. Three hundred examples have been added to the test set, and two hundred examples have been added to the training set. This division ensures that the model is evaluated on the 300 good examples during testing, providing a trustworthy assessment of its performance, considering the inaccuracies present in the non-human annotated captions.
2. The non-human annotated examples have been split into training, validation, and test sets using a split of 0.7, 0.15, and 0.15 respectively. The split is stratified based on the genre of the painting.