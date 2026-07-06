# diabetes_prediction
# Diabetes Classification Using Machine Learning

This project develops and evaluates several machine learning models for diabetes classification using the Pima Indians Diabetes dataset. The main goal is to compare different classification algorithms and improve model reliability through preprocessing, pipeline-based modeling, cross-validation, hyperparameter tuning, and threshold optimization.

## Project Overview

The dataset contains clinical features related to diabetes risk, such as glucose level, blood pressure, insulin, BMI, age, and other health-related variables. The target variable is `Outcome`, where:

* `0` represents non-diabetic cases
* `1` represents diabetic cases

Because some medical features contain biologically unrealistic zero values, these values are treated as missing data and handled using median imputation.

## Dataset

The dataset file used in this project is:

```text
diabetes.csv
```

The expected columns are:

```text
Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age
Outcome
```

Zero values in the following columns are considered biologically unrealistic and are replaced with missing values:

```text
Glucose
BloodPressure
SkinThickness
Insulin
BMI
```

## Methods

The project includes the following steps:

1. Loading and inspecting the dataset
2. Replacing unrealistic zero values with missing values
3. Splitting the dataset into training and testing sets
4. Building machine learning pipelines
5. Applying median imputation for missing values
6. Applying feature scaling where required
7. Comparing multiple classification models using stratified cross-validation
8. Selecting the best-performing model based on F1-score
9. Evaluating the selected model on an independent test set
10. Performing hyperparameter tuning
11. Optimizing the classification threshold
12. Visualizing performance using confusion matrix, ROC curve, and precision-recall curve

## Machine Learning Models

The following models are evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors
* Support Vector Machine

Pipeline-based modeling is used to ensure that preprocessing steps are applied correctly and to prevent data leakage.

## Evaluation Metrics

The models are evaluated using the following metrics:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

For this classification problem, accuracy alone is not sufficient because the dataset may be imbalanced. Therefore, F1-score, recall, and ROC-AUC are also considered important evaluation metrics.

In a medical risk-classification context, recall is especially important because failing to identify diabetic cases can be more critical than producing some false positives.

## Requirements

The project requires the following Python libraries:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
```

You can install the required packages using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run

1. Place the `diabetes.csv` file in the same folder as the notebook or Python script.

2. Run the import section first:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
```

3. Load the dataset:

```python
df = pd.read_csv("diabetes.csv")
```

4. Run the preprocessing, model training, cross-validation, and evaluation cells in order.

If using Jupyter Notebook, it is recommended to run:

```text
Kernel > Restart & Run All
```

if a `NameError` occurs. This usually means that some import or variable-definition cells were not executed in the correct order.

## Model Selection

The best model is selected based on the highest cross-validation F1-score. After selection, the model is trained on the full training dataset and evaluated on the independent test dataset.

## Hyperparameter Tuning

Random Forest is further optimized using `GridSearchCV`. The tuning process searches for the best combination of parameters such as:

* Number of trees
* Maximum tree depth
* Minimum samples required to split a node
* Minimum samples required at a leaf node

The tuned model is then evaluated on the test set.

## Threshold Optimization

The default classification threshold is 0.5. However, this threshold may not provide the best balance between precision and recall.

To improve the final classification performance, different thresholds are tested, and the threshold with the best F1-score is selected.

This step is especially useful for medical screening problems where recall may be prioritized.

## Outputs

The project generates the following outputs:

* Cross-validation comparison table
* Best model name
* Classification report
* Confusion matrix
* ROC curve
* Precision-recall curve
* Threshold comparison table
* Final model evaluation after threshold optimization

## Important Notes

This project is intended for machine learning practice and risk-classification analysis. The model should not be interpreted as a clinical diagnostic tool.

The results depend on the dataset size, feature quality, preprocessing method, and chosen evaluation metric. Further validation using larger and more diverse clinical datasets would be required before any real-world medical application.

## Suggested Project Description

This project developed a machine learning-based diabetes risk classification model using clinical features from the Pima Indians Diabetes dataset. Biologically unrealistic zero values were treated as missing values and handled using median imputation. Multiple classification models were compared using stratified cross-validation, and the best-performing model was evaluated on an independent test set. Hyperparameter tuning and threshold optimization were applied to improve classification performance, with particular attention to F1-score, recall, and ROC-AUC.

## Author

Zahra Rezaee


