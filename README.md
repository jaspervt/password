# Password Strength
This repository contains multiple python scripts that use machine learning to predict the strength of a password. The scripts use a dataset of passwords and their strengths to train different machine learning models.

## Scripts
- `modelNN.py` uses a Keras Sequential model to predict password strength
- `modelDESCT.py` uses a sklearn DecisionTreeClassifier model to predict password strength
- `modelKnn.py` uses a sklearn KNeighborsClassifier model to predict password strength
- `modelRFC.py` uses a sklearn RandomForestClassifier model to predict password strength
- `modelRFR.py` uses a sklearn RandomForestRegressor model to predict password strength
- `plot_generator.py` contains code to generate a confusion matrix, error measurements and ROC strength plots

## Dependencies
The scripts in this repository require the following libraries:
- numpy
- pandas
- keras
- sklearn

## Usage
To use the scripts, clone the repository and navigate to the directory in your terminal. Then, run the desired script using Python.

## Saved model
The ModelNN.h5 is the trained model for neural network. This model is trained with 250 epochs with a batch size of 128.