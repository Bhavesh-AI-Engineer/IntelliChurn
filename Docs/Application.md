# 🖥️ IntelliChurn Application

## 1. Application Overview

IntelliChurn is deployed as an interactive web application using **Streamlit**.

The application allows a user to enter customer information through a structured interface and receive an immediate churn prediction.

The application connects the trained V5 machine learning pipeline with a business-oriented user interface.

---

## 2. Application Workflow

The application follows this workflow:

```text
User Input
    ↓
Customer Information
    ↓
Input Validation
    ↓
Feature Preparation
    ↓
V5 Machine Learning Pipeline
    ↓
Churn Probability
    ↓
Churn / Stay Classification
    ↓
Risk Level
    ↓
Retention Recommendation
```

The application does not perform preprocessing independently.

Instead, the trained Scikit-learn pipeline handles the required preprocessing before passing the transformed data to the Random Forest classifier.

---

## 3. Customer Input Sections

The application organizes customer information into practical categories.

### Customer Profile

The user provides:

* Age
* Gender
* Married
* Family Members
* Satisfaction Score
* Tenure

### Contract & Billing

The application collects:

* Contract
* Monthly Charge
* Payment Method
* Paperless Billing

### Services

Service-related information includes:

* Internet Type
* Extra Phone Numbers
* Online Security
* Online Backup
* Device Protection
* Premium Tech Support
* Streaming TV
* Streaming Movies
* Streaming Music
* Unlimited Data

### Advanced Customer Information

Additional information can include:

* Referrals
* Total Charges
* Total Revenue
* Long Distance Charges
* Average Monthly Long Distance Charges
* Average Monthly GB Download

These inputs allow the model to consider different aspects of customer behavior.

---

## 4. Prediction

After the user submits the customer information, IntelliChurn sends the prepared input to the trained V5 pipeline.

The model produces a churn probability.

For example:

```text
Churn Probability: 73.91%
```

The application then applies the selected classification threshold.

The V5 threshold is:

```text
0.50
```

Therefore:

```text
Probability >= 0.50
        ↓
Likely to CHURN
```

while:

```text
Probability < 0.50
        ↓
Likely to STAY
```

---

## 5. Risk Classification

The application converts the prediction probability into a business-friendly risk interpretation.

A customer with a high predicted churn probability receives a higher risk classification.

This makes the prediction easier to understand for non-technical users.

Instead of displaying only:

```text
Prediction = Yes
```

the application can communicate:

```text
Churn Probability = 73.91%

Prediction = Likely to Churn

Risk Level = High
```

This makes the model output more useful from a business perspective.

---

## 6. Retention Recommendations

IntelliChurn is designed not only to identify customers at risk, but also to support potential retention decisions.

Depending on the customer's profile and predicted risk, the application can provide retention-oriented suggestions.

Possible actions include:

* Reviewing the customer's contract
* Offering a suitable loyalty incentive
* Providing a targeted service upgrade
* Reviewing pricing concerns
* Improving technical support
* Offering relevant service benefits
* Contacting high-risk customers proactively

These recommendations are intended as business guidance rather than guaranteed outcomes.

---

## 7. Model Integration

The Streamlit application loads the trained model using `joblib`.

The saved model is a complete Scikit-learn pipeline containing:

```text
Preprocessing
     +
Random Forest Classifier
```

This means the application does not need to recreate the preprocessing steps used during model training.

The application simply prepares the customer input in the expected feature format and passes it to the saved pipeline.

---

## 8. Error Handling

The application includes checks for common deployment problems.

For example, if the model file cannot be located, the application displays an informative error instead of crashing unexpectedly.

Example:

```text
❌ IntelliChurn V5 model was not found.
```

This was particularly useful during development because the project structure was changed several times while organizing the final repository.

The application was ultimately configured to locate the trained V5 model from the project's model directory.

---

## 9. User Experience

The application was designed with a business-user perspective rather than exposing raw machine learning implementation details.

The interface focuses on:

* Clear customer input fields
* Logical grouping of information
* Simple prediction results
* Churn probability
* Risk classification
* Business-oriented recommendations

The goal is to make the system understandable even to someone without a machine learning background.

---

## 10. Technology Used

The application uses:

| Technology    | Purpose                                |
| ------------- | -------------------------------------- |
| Python        | Application and machine learning logic |
| Streamlit     | Web application interface              |
| Scikit-learn  | Machine learning pipeline              |
| Random Forest | Churn classification                   |
| Pandas        | Data handling                          |
| NumPy         | Numerical operations                   |
| Joblib        | Model serialization                    |

---

## 11. Application Architecture

The overall application architecture can be represented as:

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
             Input DataFrame
                     │
                     ▼
          Saved V5 ML Pipeline
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   Preprocessing          Random Forest
          │                     │
          └──────────┬──────────┘
                     ▼
             Churn Probability
                     │
                     ▼
             Business Decision
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
       Risk Level        Recommendation
```

---

## 12. Deployment Goal

The final objective is to make IntelliChurn usable as a standalone machine learning application.

A user should not need to open a Jupyter Notebook or understand the underlying model to use the system.

The intended workflow is:

```text
Open Application
       ↓
Enter Customer Information
       ↓
Click Predict
       ↓
View Churn Probability
       ↓
View Risk Level
       ↓
Review Retention Recommendation
```

This transforms the project from a machine learning notebook into a practical **end-to-end ML application**.
