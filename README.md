# Customer Churn Data Profiler & Machine Learning Project

## Project Overview

This project performs **end-to-end Customer Churn Analysis** using:

* Multiple Data Sources (CSV, JSON, SQL, API)
* Data Cleaning & Preprocessing
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Machine Learning Models
* Feature Importance Analysis
* Automated Profiling Report
* Docker Support

The goal is to predict **Customer Churn (0 = No, 1 = Yes)** using behavioral and demographic features.

---

# Project Structure

```
├── data_profiler.py
├── customers_churn_dataset.csv
├── customers_churn_dataset.json
├── customers_churn_database.db
├── customers_churn_api_sample.json
├── Customer_Churn_Profile_Report.html
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── README.md
```

---

# Data Sources

The project loads and merges customer data from:

1. CSV Dataset
2. JSON Dataset
3. SQLite Database
4. Simulated API JSON

After loading, all datasets are merged into a single DataFrame.

# Data Cleaning Process

* Removed duplicate records
* Handled missing values:

  * Numerical → Filled with mean
  * Categorical → Filled with mode
* Dropped `CustomerID`
* Encoded categorical variables
* Created new feature:

```python
Avg_Spending = Total_Spending / Purchase_Frequency
```

---

# Exploratory Data Analysis (EDA)

The following visualizations are generated:

### Churn Distribution

Shows class imbalance between churn and non-churn customers.

### Age Distribution

Histogram with KDE curve.

### Income Distribution

Income spread across customers.

### Gender vs Churn

Categorical comparison.

### Correlation Heatmap

Shows relationships between numeric features.

### Feature Importance

From Random Forest model.

---

# Machine Learning Models Used

## Logistic Regression

* Scaled features using StandardScaler
* Binary classification

## Random Forest Classifier

* 200 estimators
* Feature importance extracted

---

# Model Evaluation Metrics

* Accuracy Score
* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)

---

# Key Insight

From Feature Importance graph:

* **Support_Calls_Last_Year** is the most influential feature
* Engagement and Discount usage also impact churn
* Demographics (Age, Gender) have lower impact

---

# Automated Profiling Report

A full profiling report is generated using:

* `ydata-profiling`

Output file:

```
Customer_Churn_Profile_Report.html
```

Open it in browser to see:

* Missing values
* Correlations
* Data types
* Statistical summaries
* Distributions

---

# Installation

## Clone Repository

```bash
git clone <repo-url>
cd customer-churn-project
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* ydata-profiling

---

# Run Project

```bash
python data_profiler.py
```

You will see:

* Dataset merging logs
* Cleaning summary
* Model accuracy results
* Multiple visualizations
* Feature importance graph

---

# Docker Support

To build and run using Docker:

```bash
docker build -t churn-profiler .
docker run churn-profiler
```

---

# Technologies Used

* Python 3
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* SQLite
* YData Profiling
* Docker

---

# Business Objective

This system helps businesses:

* Identify high-risk churn customers
* Understand churn-driving factors
* Improve retention strategy
* Optimize customer engagement

---

# Example Features in Dataset

* Age
* Gender
* City
* Income
* Purchase Frequency
* Total Spending
* Last Purchase Days Ago
* Customer Tenure
* Support Calls
* Discount Usage Rate
* Engagement Score
* Churn (Target)
