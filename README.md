Neural Artwork Caption Generator
==============================

This repository contains the code used in the paper ["Exploring the Synergy Between Vision-Language Pretraining and ChatGPT for Artwork Captioning: A Preliminary Study"](https://www.researchgate.net/publication/377558832_Exploring_the_Synergy_Between_Vision-Language_Pretraining_and_ChatGPT_for_Artwork_Captioning_A_Preliminary_Study).

![Overview](https://cdn.gamma.app/8sbvs4icoang31g/e07b9c6ebc564bf5ad924115b7eb9c13/original/Screenshot-2023-09-24-170020.png)

The system is designed for artwork captioning: given the input image of an artwork, the neural network produces an extensive description based on the results of a multi-task classification module
and of a fine-tuned vision captioning model.

Project Organization
------------

We are currently refactoring the repository to facilitate the reproducbility of the results. Stay tuned!

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

Authors and Acknowlodgements
----------------------------

This repo is part of Nicola Fanelli's MSc thesis in Deep Learning, realized with the supervision of Prof. Gennaro Vessio, Prof. Giovanna Castellano and Raffaele Scaringi.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
