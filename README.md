# 🧠 IntelliChurn

### End-to-End Customer Churn Prediction & Retention System

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Project-Completed-success)]()

---

## 📌 Overview

**IntelliChurn** is an end-to-end machine learning application designed to predict whether a telecom customer is likely to churn.

The system analyzes customer demographic, contractual, billing, usage, and service-related information and produces:

* 🎯 Churn / Stay prediction
* 📊 Churn probability
* ⚠️ Customer risk level
* 💡 Recommended retention actions

The final system uses a **Random Forest Classifier** integrated into a Scikit-learn preprocessing pipeline and deployed through an interactive **Streamlit web application**.

---

## 🎯 Business Problem

Customer churn is a major challenge for subscription-based businesses.

Instead of waiting for customers to leave, companies can use predictive analytics to identify customers who are more likely to churn and prioritize retention efforts.

IntelliChurn addresses the question:

> **How can a telecom company identify customers who are likely to churn before they actually leave?**

The system goes beyond a simple Yes/No prediction by providing a probability score and business-oriented risk interpretation.

### Example

| Customer   | Churn Probability | Interpretation |
| ---------- | ----------------: | -------------- |
| Customer A |             2.00% | Very stable    |
| Customer B |            18.39% | Low risk       |
| Customer C |            73.91% | High risk      |

This allows businesses to move from **reactive retention** toward **proactive customer retention**.

---

## ✨ Key Features

* 🔍 Exploratory data analysis
* 🧹 Structured data preprocessing
* 🧩 Automated feature transformation
* 🌲 Random Forest classification
* 📈 ROC-AUC evaluation
* 🎯 Churn probability prediction
* ⚖️ Prediction threshold analysis
* ⚠️ Customer risk classification
* 💡 Retention recommendations
* 🖥️ Interactive Streamlit interface
* 🔄 End-to-end ML pipeline
* 📦 Serialized production model

---

## 📊 Dataset

The project uses a telecom customer churn dataset containing:

* **7,043 customer records**
* **41 original columns**

The target variable is:

```text
Churn Label
```

with:

```text
No  → Customer stayed
Yes → Customer churned
```

### Target Distribution

| Churn Label |     Count |
| ----------- | --------: |
| No          |     5,174 |
| Yes         |     1,869 |
| **Total**   | **7,043** |

The final train/test split contains:

| Churn Label | Training | Testing |
| ----------- | -------: | ------: |
| No          |    4,139 |   1,035 |
| Yes         |    1,495 |     374 |

---

## 🧩 Feature Engineering & Preprocessing

The final V5 model uses **36 input features**:

```text
22 categorical features
14 numerical features
---------------------
36 total features
```

The preprocessing and model are packaged together inside a Scikit-learn `Pipeline`.

This ensures that the same transformation logic used during training is applied during prediction.

### Important Feature Validation

The final V5 model was explicitly validated to ensure that:

```text
Input feature count = 36
Satisfaction Score = NOT used by V5 model
```

The Streamlit interface may display additional customer information for user context, but the V5 prediction pipeline only receives its validated 36-feature model input.

This distinction prevents training/inference feature mismatches.

---

## 🤖 Final Model

### Random Forest Classifier

```text
Algorithm:
Random Forest Classifier

n_estimators:
200

random_state:
42
```

The classifier is contained inside the final Scikit-learn pipeline.

### Model Development

Multiple model versions were developed and evaluated before selecting the V5 candidate.

The development process considered:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrix
* Overfitting
* Prediction threshold

See the detailed documentation:

📘 **[Model Development](Docs/Model%20Development.md)**

---

## 🏆 V5 Performance

The final V5 model achieved:

| Metric          |      Score |
| --------------- | ---------: |
| Accuracy        | **82.82%** |
| ROC-AUC         | **90.31%** |
| Churn Precision | **63.58%** |
| Churn Recall    | **82.62%** |
| Churn F1-score  | **71.86%** |

### Confusion Matrix

```text
                 Predicted
                 No      Yes

Actual No        858     177
Actual Yes        65     309
```

The model correctly identified:

> **309 out of 374 churned customers**

This produces a churn recall of approximately:

> **82.62%**

For a retention-focused system, identifying a large proportion of actual churners is particularly valuable.

---

## 🎚️ Prediction Threshold

The V5 probability threshold was evaluated across multiple values.

| Threshold |  Precision |     Recall |   F1 Score |
| --------: | ---------: | ---------: | ---------: |
|      0.30 |     0.5021 |     0.9572 |     0.6587 |
|      0.35 |     0.5364 |     0.9251 |     0.6791 |
|      0.40 |     0.5607 |     0.8770 |     0.6840 |
|      0.45 |     0.5966 |     0.8503 |     0.7012 |
|  **0.50** | **0.6358** | **0.8262** | **0.7186** |
|      0.55 |     0.6629 |     0.7781 |     0.7159 |
|      0.60 |     0.6865 |     0.7086 |     0.6974 |

The highest F1-score occurred at:

```text
Threshold = 0.50
```

Therefore, IntelliChurn V5 uses:

```text
Probability >= 0.50
        ↓
Likely to CHURN

Probability < 0.50
        ↓
Likely to STAY
```

---

## 🖥️ Streamlit Application

The trained V5 pipeline is integrated into an interactive Streamlit application.

The application allows users to:

1. Enter customer information
2. Submit the customer profile
3. Generate a churn probability
4. View the churn/stay prediction
5. View the customer's risk level
6. Review recommended retention actions

### Application Output

For example:

```text
Churn Probability: 73.91%

Prediction:
Likely to CHURN

Risk Level:
High Risk
```

A lower-risk customer might produce:

```text
Churn Probability: 18.39%

Prediction:
Likely to STAY

Risk Level:
Low Risk
```

Detailed application documentation:

📘 **[Application Documentation](Docs/Application.md)**

---

## 🏗️ Architecture

```text
                    Customer Dataset
                           │
                           ▼
                  Exploratory Analysis
                           │
                           ▼
                    Data Preparation
                           │
                           ▼
                  Feature Processing
                           │
                           ▼
                 Random Forest V5
                           │
                           ▼
                   Saved ML Pipeline
                           │
                           ▼
                  Streamlit Application
                           │
                           ▼
                  Customer Information
                           │
                           ▼
                  Churn Probability
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
              Risk Level       Recommendation
```

---

## 🛠️ Technology Stack

| Technology       | Purpose                            |
| ---------------- | ---------------------------------- |
| Python           | Programming language               |
| Pandas           | Data manipulation                  |
| NumPy            | Numerical operations               |
| Scikit-learn     | Machine learning and preprocessing |
| Random Forest    | Churn classification               |
| Streamlit        | Web application                    |
| Joblib           | Model serialization                |
| Matplotlib       | Data visualization                 |
| Seaborn          | Exploratory visualization          |
| Jupyter Notebook | Model experimentation              |
| Git              | Version control                    |

---

## 📁 Project Structure

```text
IntelliChurn/
│
├── App/
│   ├── app.py
│   └── Models/
│       ├── intellichurn_model_v5.pkl
│       └── threshold.txt
│
├── Data/
│
├── Images/
│
├── Models/
│   ├── intellichurn_model.pkl
│   └── intellichurn_v4.pkl
│
├── Notebooks/
│   ├── 01_data_exploration.ipynb
│   └── Model_Training.ipynb
│
├── Docs/
│   ├── Application.md
│   ├── Model Development.md
│   ├── Project Overview.md
│   └── Project Structure.md
│
├── Reports/
│
├── Src/
│
├── README.md
├── requirements.txt
└── .gitignore
```

Detailed project structure:

📘 **[Project Structure](Docs/Project%20Structure.md)**

---

## 🚀 Running the Application

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd IntelliChurn
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

#### Windows

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run Streamlit

```bash
streamlit run App/app.py
```

The application will open in your browser.

---

## 📚 Documentation

More detailed documentation is available in the `Docs` directory.

| Document                                         | Description                                                                               |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| [Project Overview](Docs/Project%20Overview.md)   | Business problem, objectives, dataset, preprocessing, EDA and overall project explanation |
| [Model Development](Docs/Model%20Development.md) | Model experimentation, evaluation, V5 selection and threshold analysis                    |
| [Application](Docs/Application.md)               | Streamlit application, prediction workflow and model integration                          |
| [Project Structure](Docs/Project%20Structure.md) | Repository organization and file responsibilities                                         |

---

## 🔮 Future Improvements

Potential future improvements include:

* 🔬 Hyperparameter optimization
* 📊 Model explainability using SHAP
* 📈 Interactive analytics dashboard
* 🎯 Customer segmentation
* 💰 Estimated revenue-at-risk analysis
* 🔄 Automated model retraining
* ☁️ Cloud deployment
* 📡 Monitoring model performance after deployment
* 🧠 Comparison with gradient boosting models
* 🔐 Production-grade input validation

---

## 🎓 Learning Outcomes

Through IntelliChurn, the project demonstrates practical experience with:

* End-to-end machine learning workflows
* Classification problems
* Feature preprocessing
* Categorical encoding
* Random Forest models
* Model evaluation
* ROC-AUC analysis
* Precision/recall trade-offs
* Threshold optimization
* Overfitting analysis
* Model serialization
* Streamlit deployment
* Business-oriented ML interpretation

---

## 👨‍💻 Author

**Bhavesh Pawar**

B.Tech Artificial Intelligence & Data Science

---

## ⭐ Project Status

**IntelliChurn V5 is the current production candidate used by the Streamlit application.**

The model, preprocessing pipeline, prediction threshold, application interface, and documentation have been validated as part of the final project workflow.

If you found this project interesting, feel free to explore the documentation and implementation.
