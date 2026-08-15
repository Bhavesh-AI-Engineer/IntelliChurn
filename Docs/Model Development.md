# 🤖 IntelliChurn Model Development

## 1. Model Development Strategy

IntelliChurn was developed iteratively rather than relying on a single model configuration.

Multiple Random Forest versions were trained and evaluated to understand how different feature selections and model configurations affected churn prediction performance.

The main goal was not simply to maximize accuracy.

Instead, the model was evaluated with particular attention to:

* Churn recall
* Churn precision
* F1-score
* ROC-AUC
* Confusion matrix
* Overfitting
* Prediction threshold

This was important because IntelliChurn is a **customer retention system**, where failing to identify a customer who is actually going to churn can be more costly than incorrectly flagging a stable customer.

---

## 2. Dataset Split

The dataset contains **7,043 customers**.

The data was divided into training and testing sets using an **80/20 split**.

```text
Total samples:       7043
Training samples:    5634
Testing samples:     1409
```

The target distribution was:

| Churn Label | Training | Testing |
| ----------- | -------: | ------: |
| No          |    4,139 |   1,035 |
| Yes         |    1,495 |     374 |

The test set was kept separate from model training so that the final evaluation represented the model's performance on unseen customer records.

---

## 3. Feature Configuration

The final V5 model uses **36 input features**.

```text
Categorical features: 22
Numerical features:   14
Total features:       36
```

The feature structure was explicitly validated before training.

One important validation was confirming that:

```text
Satisfaction Score
```

was **not present** in the final V5 input features.

This prevented a mismatch between the model's preprocessing configuration and the actual dataset.

---

## 4. Preprocessing Pipeline

The model uses a Scikit-learn `Pipeline` combined with a `ColumnTransformer`.

Categorical and numerical variables are processed separately.

### Categorical Features

Categorical variables are transformed using:

```text
OneHotEncoder(
    drop="first",
    handle_unknown="ignore"
)
```

This converts categorical values into numerical representations that can be processed by the Random Forest classifier.

`handle_unknown="ignore"` is particularly useful during deployment because the application may encounter a category that was not present in the training data.

### Numerical Features

Numerical features are passed through the numerical preprocessing stage before being supplied to the classifier.

The preprocessing logic is stored inside the model pipeline itself.

Therefore:

```text
Training Data
     ↓
Preprocessing
     ↓
Random Forest
```

and:

```text
Streamlit Input
     ↓
Same Preprocessing
     ↓
Same Random Forest
```

use the same transformation logic.

This significantly reduces the possibility of training/inference inconsistencies.

---

## 5. Model Versions

Several model versions were developed during the project.

### V3

V3 achieved approximately:

```text
Accuracy: 82.40%
ROC-AUC:  87.38%
```

The model was useful but showed weaker overall discrimination than later versions.

---

### V4

V4 improved the model substantially:

```text
Accuracy: 84.39%
ROC-AUC:  90.00%
```

The churn class achieved:

```text
Precision: 69%
Recall:    75%
F1-score:  72%
```

V4 demonstrated that the model could identify a significantly larger proportion of churned customers.

---

### V5

V5 focused more heavily on churn detection.

Final performance:

```text
Accuracy: 82.82%
ROC-AUC:  90.31%
```

Churn performance:

```text
Precision: 63.58%
Recall:    82.62%
F1-score:  71.86%
```

Although V5 had slightly lower accuracy than V4, it achieved a higher churn recall.

For IntelliChurn, this trade-off is valuable because the application is intended to identify customers who may leave and allow the business to intervene.

---

## 6. Final V5 Model

The final model uses:

```text
Algorithm:
Random Forest Classifier

Number of Trees:
200

Random State:
42
```

The classifier is packaged inside the complete preprocessing pipeline.

Conceptually:

```text
Customer Data
      ↓
ColumnTransformer
      ↓
Categorical Encoding
      ↓
Numerical Processing
      ↓
Random Forest
      ↓
Churn Probability
```

The complete pipeline is saved as a `.pkl` file using `joblib`.

This allows the trained model to be loaded directly by the Streamlit application without rebuilding the preprocessing steps.

---

## 7. Model Evaluation

The final V5 model produced the following results:

| Metric          | Result |
| --------------- | -----: |
| Accuracy        | 82.82% |
| ROC-AUC         | 90.31% |
| Churn Precision | 63.58% |
| Churn Recall    | 82.62% |
| Churn F1-score  | 71.86% |

### Confusion Matrix

```text
                  Predicted
                  No      Yes

Actual No        858     177
Actual Yes        65     309
```

The model correctly identified:

```text
309 / 374
```

customers who actually churned.

This corresponds to a churn recall of approximately:

```text
82.62%
```

In a retention-oriented application, this is an important metric because it measures how many actual churners the model successfully identifies.

---

## 8. ROC-AUC

The final V5 model achieved:

```text
ROC-AUC = 0.9031
```

ROC-AUC measures how effectively the model separates customers who churn from customers who stay across different classification thresholds.

A value close to `1.0` indicates strong discrimination.

The V5 result of approximately `0.90` indicates that the model has strong ability to distinguish between the two customer groups.

---

## 9. Overfitting Analysis

An overfitting check was also performed by comparing training and testing performance.

V5 produced:

```text
Training Accuracy : 89.12%
Testing Accuracy  : 82.82%

Training ROC-AUC  : 97.33%
Testing ROC-AUC   : 90.31%
```

The resulting gaps were:

```text
Accuracy Gap : 6.29%
ROC-AUC Gap  : 7.03%
```

This indicates that the model performs better on training data than unseen testing data, which is expected for a Random Forest.

However, the testing ROC-AUC remains strong at approximately `0.90`, suggesting that the model retains useful predictive capability on unseen customers.

---

## 10. Prediction Threshold

The model produces a churn probability.

A classification threshold determines when that probability becomes a `Yes` prediction.

Several thresholds were evaluated:

| Threshold |  Precision |     Recall |   F1 Score |
| --------: | ---------: | ---------: | ---------: |
|      0.30 |     0.5021 |     0.9572 |     0.6587 |
|      0.35 |     0.5364 |     0.9251 |     0.6791 |
|      0.40 |     0.5607 |     0.8770 |     0.6840 |
|      0.45 |     0.5966 |     0.8503 |     0.7012 |
|  **0.50** | **0.6358** | **0.8262** | **0.7186** |
|      0.55 |     0.6629 |     0.7781 |     0.7159 |
|      0.60 |     0.6865 |     0.7086 |     0.6974 |

The highest F1-score was obtained at:

```text
Threshold = 0.50
F1-score  = 0.7186
```

Therefore, IntelliChurn uses:

```text
Probability >= 0.50
        ↓
     CHURN

Probability < 0.50
        ↓
      STAY
```

The threshold can be changed in the application if a business wants to prioritize recall or precision differently.

---

## 11. Feature Importance

Feature importance was analyzed to understand which variables contributed most strongly to the Random Forest's predictions.

Among the most important features in V5 were:

| Feature                               | Importance |
| ------------------------------------- | ---------: |
| Tenure in Months                      |     0.0801 |
| Number of Referrals                   |     0.0800 |
| Contract - Two Year                   |     0.0795 |
| Monthly Charge                        |     0.0738 |
| Total Revenue                         |     0.0733 |
| Total Charges                         |     0.0681 |
| Total Long Distance Charges           |     0.0582 |
| Age                                   |     0.0502 |
| Average Monthly GB Download           |     0.0501 |
| Average Monthly Long Distance Charges |     0.0463 |

These results suggest that **tenure, contract structure, customer engagement, and billing-related variables** play an important role in the model's predictions.

However:

> Feature importance indicates predictive usefulness, not causation.

For example, a high importance for tenure does not prove that increasing or decreasing tenure directly causes churn.

---

## 12. Why V5 Was Selected

V5 was selected as the final model because it provides a strong balance between predictive performance and the business objective of identifying potential churners.

Compared with earlier versions, V5 achieved:

* Strong ROC-AUC
* High churn recall
* Competitive F1-score
* Reasonable precision
* Acceptable generalization to unseen data
* A clearly evaluated prediction threshold

The final model therefore prioritizes **useful churn detection rather than accuracy alone**.

---

## 13. Model Deployment

The trained V5 pipeline is integrated into the Streamlit application.

The deployment architecture is:

```text
                    IntelliChurn
                         │
                         ▼
                Streamlit Interface
                         │
                         ▼
                 Customer Inputs
                         │
                         ▼
              V5 ML Pipeline
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        Churn Probability      Prediction
              │                     │
              └──────────┬──────────┘
                         ▼
                  Risk Assessment
                         │
                         ▼
              Retention Recommendation
```

The application loads the saved model and uses it to generate predictions for newly entered customer information.
