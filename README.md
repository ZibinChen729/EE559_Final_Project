# Heart Disease Prediction Using Supervised Learning Methods

## Project Overview

This project predicts whether a patient has heart disease using supervised learning methods. The task is a binary classification problem, where the target variable is:

- `0`: no heart disease
- `1`: heart disease

Three supervised learning models are implemented and compared:

- Logistic Regression
- Decision Tree
- Multi-Layer Perceptron

The models are evaluated using accuracy, precision, recall, F1-score, confusion matrices, and learning curves.

---

## Dataset

The dataset used in this project is the public UCI Heart Disease dataset from Kaggle.

The dataset file used in this project is:

`HeartDiseaseTrain-Test.csv`

The original dataset contains:

- 1,025 samples
- 14 columns
- 13 input features
- 1 binary target variable

During preprocessing, duplicated records were removed. A total of 723 duplicated rows were removed, leaving 302 unique samples for model training and evaluation.

---

## Features

The dataset contains both numerical and categorical features.

Numerical features include:

- age
- resting blood pressure
- cholesterol
- maximum heart rate
- oldpeak

Categorical features include:

- sex
- chest pain type
- fasting blood sugar
- resting ECG result
- exercise-induced angina
- slope
- number of major vessels
- thalassemia

---

## Preprocessing

The preprocessing steps include:

- loading the dataset
- checking missing values
- removing duplicated rows
- resetting the index after removing duplicates
- separating input features and target variable
- standardizing numerical features for Logistic Regression and Multi-Layer Perceptron
- applying one-hot encoding to categorical features
- keeping numerical features unscaled for Decision Tree

Logistic Regression and Multi-Layer Perceptron use `StandardScaler` because these models are sensitive to feature scale. Decision Tree does not require numerical scaling.

---

## Models

### Logistic Regression

Logistic Regression was used as the baseline model. The regularization parameter `C` was tuned using GridSearchCV. The tested values were:

- `0.01`
- `0.1`
- `1`
- `10`
- `100`

The solver used was `lbfgs`, and the maximum number of iterations was set to `1000`.

### Decision Tree

Decision Tree was used because it is easy to understand and can learn nonlinear patterns. To reduce overfitting, the following hyperparameters were tuned:

- maximum depth
- minimum samples split
- minimum samples leaf
- splitting criterion

The tested criteria were:

- `gini`
- `entropy`

### Multi-Layer Perceptron

Multi-Layer Perceptron was used to test whether a neural network model could improve prediction performance. The following settings were tested:

- hidden layer sizes: `(8)`, `(16)`, `(16, 8)`, `(32)`
- activation functions: `relu`, `tanh`
- alpha values: `0.001`, `0.01`, `0.1`
- learning rate: `0.001`

Early stopping and L2 regularization were used to reduce overfitting.

---

## Experiment Setup

The dataset was split into:

- 60% training data
- 20% validation data
- 20% test data

Stratified splitting was used to keep the class distribution similar across the training, validation, and test sets.

Hyperparameter tuning was performed using:

`GridSearchCV with 5-fold cross-validation`

The main scoring metric was:

`F1-score`

F1-score was used because both precision and recall are important for this binary classification task.

---

## Results

The final test results are shown below:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8361 | 0.8485 | 0.8485 | 0.8485 |
| Decision Tree | 0.7705 | 0.7714 | 0.8182 | 0.7941 |
| Multi-Layer Perceptron | 0.8197 | 0.8929 | 0.7576 | 0.8197 |

Logistic Regression achieved the best overall F1-score. Multi-Layer Perceptron achieved the highest precision, but its recall was lower than the other models.

---

## Output

The notebook produces:

- dataset information
- missing value summary
- target distribution
- data visualization plots
- best hyperparameters for each model
- final test performance table
- confusion matrices
- final model comparison bar chart
- training vs validation error curves

---

## How to Run

1. Clone or download this repository.
2. Make sure `HeartDiseaseTrain-Test.csv` is in the same folder as the notebook.
3. Open `heart_disease_prediction.ipynb` using Jupyter Notebook or Google Colab.
4. Run all cells from top to bottom.
