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

## Key results

- Random forest classifier achieved a test F1 score of 0.76.
- Hyperparameter tuning reduced overfitting and improved metrics compared to the default model.
- MLP regression achieved a test RMSE of 0.11.
- Most influential variables included Co, Ni, Se, V, and Time.
- Model performance remained consistent across train, validation, and test datasets, suggesting reasonable generalization.

## Dataset

- Experimental electrocatalysis data obtained from a chemistry research group at UCLA
- 545 samples, 13 features
- Target variables: overpotential value (used in regression with MLP) and overpotential classification as good or bad (used in classification with random forest)

# Description

## Part 1: Preparing Training, Validation, and Test Splits

First, training, validation, and test splits are created using 70% of the data for training, 15% for validation, and 15% for final testing. Data is shuffled so that the split is randomized rather than dependent on the order of the original CSV file.

The same process is repeated with a second split that uses more data for training: 80% training, 15% validation, and 5% testing for comparison of split sizes on model behavior.

## Part 2: Random Forest Classification

For the classification task, catalysts with `|overpotential| < 0.6` are defined as **good** and catalysts with `|overpotential| >= 0.6` as **bad**. The goal is to predict whether a catalyst is good or bad from the experimental variables.

5-fold cross-validation was used to evaluate model performance based on training and validation data. Precision, recall, and F1 scores quantitatively indicate how well the model can classify catalysts. 

Initially, a default random forest model was trained. Evidence of overfitting was observed because cross-validation training metrics were substantially higher than validation metrics (shown below).

Training Set Metrics:
Precision : Mean = 1.00, Std = 0.00
Recall    : Mean = 1.00, Std = 0.00
F1        : Mean = 1.00, Std = 0.00

Validation Set Metrics:
Precision : Mean = 0.74, Std = 0.06
Recall    : Mean = 0.66, Std = 0.09
F1        : Mean = 0.69, Std = 0.04

 ================================

To understand why the random forest was overfitting, one representative tree from the default model was plotted and the tree structure was used to guide hyperparameter tuning.

<img width="1934" height="1252" alt="image" src="https://github.com/user-attachments/assets/e0bbd54c-e55c-43eb-b778-7a96f9480555" />

Hyperparameters including `max_depth`, `min_samples_split`, and `min_samples_leaf` were altered, as the representative tree for the base model is quite deep, and this model creates leaves and branches that include very few data points. Other hyperparameters, such as the number of trees, as well as `max_features` were altered as well. Hyperparameters were selected through iterative experimentation informed by model diagnostics and tree visualization rather than exhaustive grid search. The results from the best set of chosen hyperparameters for modeling is shown below.

Training Set Metrics:
Precision : Mean = 0.81, Std = 0.02
Recall    : Mean = 0.75, Std = 0.02
F1        : Mean = 0.78, Std = 0.02

Validation Set Metrics:
Precision : Mean = 0.73, Std = 0.04
Recall    : Mean = 0.66, Std = 0.05
F1        : Mean = 0.69, Std = 0.03

 ================================

A representative tree is also shown.

<img width="1880" height="1252" alt="image" src="https://github.com/user-attachments/assets/96134db3-1376-4943-87a0-80994cc6cfd9" />

Cross-validation results indicated that all tested hyperparameter configurations perform similarly on unseen validation data. All tuned models reduced the gap between training and validation performance compared to the base model, but improvements in validation metrics were relatively small. This suggests that model performance is relatively robust to moderate changes in the tested hyperparameters.

The final model was selected because validation metrics were slightly better than those of the base model while also exhibiting a smaller gap between training and validation performance. This suggests that this model may provide a slightly better balance between predictive performance and generalization.

Validation metrics for this model were also marginally better than those with other sets of manually chosen hyperparameters. From here, test set metrics were obtained using the final chosen model (shown below).

Test Set Metrics:
Precision = 0.83
Recall = 0.70
F1 = 0.76

Metrics from the test set predictions were relatively similar to those of the validation and training sets. This suggests reduced overfitting relative to the baseline model and indicates improved generalization performance. Metrics leave substantial room for improvement, however, which may be done through further tuning of hyperparameters or using other classification models, such as gradient boosting machines (e.g., XGBoost).

Using the selected random forest hyperparameters, the 5-fold cross-validation workflow was repeated with the second data split to see how model performance changed when more data was used for training. The model performed similarly against the smaller test set in the second split compared to the first split, with a very minor improvement seen through the F1 score. However, the smaller test set also makes the test metrics less stable, so this comparison should be interpreted very carefully rather than treated as definitive proof that the second split gives a better model.

The slightly improved performance may be related to the larger training set, although differences should be interpreted cautiously due to the smaller test set.

Test Set Metrics:
Precision = 0.70
Recall = 0.88
F1 = 0.78

## Part 3: MLP Regression

For the regression task, the overpotential value was predicted directly using multilayer perceptron models. Performance was compared with and without input normalization, then permutation feature importances were used to interpret which variables matter most.

Five-fold cross-validation was used to evaluate model performance based on training and validation data. MSE and RMSE values quantitatively indicate how well the model predicted catalyst overpotential values.

Training Set Metrics:
MSE: Mean = 0.01, Std = 0.00
RMSE: Mean = 0.11, Std = 0.00

Validation Set Metrics:
MSE: Mean = 0.03, Std = 0.00
RMSE: Mean = 0.16, Std = 0.01

The 5 most important features determined using the MLP base model were Co, Ni, Se, P, and Voltage.

From here, four MLP hyperparameter combinations were tested and cross-validation performance was compared to develop an improved model. The results from the final chosen model are shown below.

Training Set Metrics:
MSE: Mean = 0.01, Std = 0.00
RMSE: Mean = 0.10, Std = 0.00

Validation Set Metrics:
MSE: Mean = 0.01, Std = 0.00
RMSE: Mean = 0.12, Std = 0.01

 ================================

This model appeared to be best because the MSE and RMSE values were relatively low while remaining reasonably comparable across the training and validation sets. Test set metrics were then calculated for this model, and permutation feature importances were used to determine the five most important features influencing overpotential values. These results are shown below.

Test Set Metrics:
MSE = 0.01
RMSE = 0.11

The 5 most important features based on the final model for MLP were Co, Se, Ni, V, and Time.

The top features from the final tuned MLP were different from the default-model rankings in the earlier PFI analysis. This was expected because hyperparameter tuning changes how the model fits the data, and scaling changes how the model handles variables with different numerical ranges.

The feature importance results should therefore be interpreted as model-dependent rather than as one universal ranking of chemical importance. PFI is useful here because it shows which features are most influential for a specific trained model, but the ranking can shift when the model architecture, training behavior, or preprocessing changes.

The final model gives somewhat low and consistent errors over all data splits (including the test set), indicating minimal overfitting. Therefore, this appears to be a reasonable model choice for catalyst overpotential prediction within this hydrogen evolution reaction dataset. To improve performance, other ML models may be tested, and hyperparameters may be further tuned, as only one type of neural network and a limited number of hyperparameter configurations were tested. 

This model should not be generalized beyond the specific chemical reaction used to generate the initial data.
