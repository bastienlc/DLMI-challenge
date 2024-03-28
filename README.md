# DLMI-challenge

[This repository](https://github.com/bastienlc/DLMI-challenge) contains our work for the challenge of the DLMI course at ENS Paris-Saclay where [@JoachimCOLLIN](https://github.com/JoachimCOLLIN) and I reached a score of 0.0.89090.

The goal of the challenge was to predict the reactive or malignant nature of lymphocytosis in patients.

## Entrypoints

* [Baseline training](baseline.py)
* [Mixture of experts training](mixture_of_experts.py)
* [Mixture of experts cross validation](cross_validation.py)
* [Unsupervised classification](unsupervised_classification.ipynb)

Note that the SCAN code to create the unsuperivsed classes is not included in this repository because the model was not used in the end and the code is pretty consequent.


## Project structure

```
.
├── README.md
├── baseline.py
├── mixture_of_experts.py
├── cross_validation.py
├── unsupervised_classification.ipynb
├── visualization.ipynb
├── requirements.txt
├── data
│   ├── test_class.csv # Unsupervised classes for the test set
│   ├── train_class.csv # Unsupervised classes for the train set
├── report # Report for the project
├── src
│   ├── datasets
│   │   ├── per_image.py # Dataset for the baseline model
│   │   ├── per_patient.py # Dataset for the mixture of experts model
│   ├── models
│   │   ├── baseline.py # Baseline model
│   │   ├── mixture_of_experts.py # Mixture of experts model
│   ├── config.py # Configuration file
│   ├── utils.py
│   ├── train.py # Training loop
│   ├── data.py # Data preprocessing and loaders
│   ├── cross_validation.py # Cross validation for unsupervised classification
│   ├── style.py # Style for the plots
```

## Contributors

[@bastienlc](https://github.com/bastienlc),
[@JoachimCOLLIN](https://github.com/JoachimCOLLIN)