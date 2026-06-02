# Electrocatalyst_Performance_Prediction
This project models electrocatalytic activity for the hydrogen evolution reaction. The workflow compares classification with random forests and regression with multilayer perceptrons (MLPs), while using train/validation/test splits, cross-validation, hyperparameter tuning, and permutation feature importance to evaluate model behavior.

## Objectives:

- Split data into training, validation, and testing sets using train_test_split
- Build random forest models and tune hyperparameters
- Build multilayer perceptron models and tune hyperparameters
- Cross-validate model performance using 5-fold cross-validation
- View and analyze decision trees that make up the chosen random forest models
- Compare MLP models using normalized vs. non-normalized data
- Critically evaluate model performance and select a final model with the most promising metrics for accurate, generalizable predictions

## Tools and libraries

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib

# Description

## Part 1: Preparing Training, Validation, and Test Splits

First, training, validation, and test splits are created using 70% of the data for training, 15% for validation, and 15% for final testing. Data is shuffled so that the split is randomized rather than dependent on the order of the original CSV file.

The same process is repeated with a second split that uses more data for training: 80% training, 15% validation, and 5% testing for comparison of split sizes on model behavior.

## Part 2: Random Forest Classification

For the classification task, catalysts with `|overpotential| < 0.6` are defined as **good** and catalysts with `|overpotential| >= 0.6` as **bad**. The goal is to predict whether a catalyst is good or bad from the experimental variables.

Feature scaling is not applied because random forests are insensitive to feature magnitude.

