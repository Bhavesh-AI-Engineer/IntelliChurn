# 📁 IntelliChurn Project Structure

The IntelliChurn project is organized into separate directories for application code, datasets, notebooks, trained models, and documentation.

```text
IntelliChurn/
│
├── App/
│   ├── app.py
│   │
│   └── Models/
│       └── intellichurn_model_v5.pkl
│
├── Data/
│   └── telco.csv
│
├── Notebooks/
│   ├── EDA.ipynb
│   ├── Model_Development.ipynb
│   └── ...
│
├── Model Development.md
├── Application.md
├── Project Structure.md
├── README.md
│
├── requirements.txt
└── .gitignore
```

## 📂 Directory Explanation

### `App/`

Contains the Streamlit application responsible for running IntelliChurn.

```text
App/
└── app.py
```

`app.py` contains the application interface, input handling, prediction logic, risk classification, and retention recommendations.

---

### `App/Models/`

Contains the trained machine learning model used by the application.

```text
App/
└── Models/
    └── intellichurn_model_v5.pkl
```

The V5 model is stored as a serialized Scikit-learn pipeline using Joblib.

The pipeline contains both:

* Feature preprocessing
* Random Forest classifier

This allows the application to use the same preprocessing logic that was used during model training.

---

### `Data/`

Contains the dataset used during the project.

The primary dataset contains telecom customer information used for exploratory data analysis and model development.

The raw dataset should not contain sensitive or personally identifiable customer information.

---

### `Notebooks/`

Contains the Jupyter Notebooks used during the development process.

The notebooks document activities such as:

* Data exploration
* Data cleaning
* Feature analysis
* Model experimentation
* Model evaluation
* Feature importance analysis
* Threshold analysis
* Overfitting checks

This directory represents the experimentation and development side of the project.

---

### `Model Development.md`

Documents the machine learning development process.

It explains how IntelliChurn progressed through different model versions and how the final V5 model was evaluated.

---

### `Application.md`

Documents the Streamlit application and explains how the trained model is integrated into the user-facing system.

---

### `README.md`

Acts as the main entry point for the project.

It provides an overview of IntelliChurn and links the major project documentation sections together.

---

### `requirements.txt`

Contains the Python dependencies required to run the project.

Typical dependencies include:

```text
pandas
numpy
scikit-learn
streamlit
joblib
matplotlib
seaborn
```

The exact versions should match the environment used to develop and test the application.

---

### `.gitignore`

Specifies files and folders that should not be committed to version control.

This can include:

```text
.venv/
__pycache__/
.ipynb_checkpoints/
*.pyc
```

Environment-specific and temporary files should generally remain outside the Git repository.

---

## 🔄 Project Flow

The overall project can be viewed as:

```text
Dataset
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Feature Preparation
   │
   ▼
Model Development
   │
   ▼
V1 → V2 → V3 → V4 → V5
   │
   ▼
Final V5 Pipeline
   │
   ▼
Saved Model
   │
   ▼
Streamlit Application
   │
   ▼
Customer Churn Prediction
```

This structure separates experimentation from deployment while keeping the final application simple and reproducible.
