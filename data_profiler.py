# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import json
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")

# 2. DATA ACQUISITION

print("Loading CSV...")
df_csv = pd.read_csv("customers_churn_dataset.csv")

print("Loading JSON...")
df_json = pd.read_json("customers_churn_dataset.json")

print("Loading SQL Database...")
conn = sqlite3.connect("customers_churn_database.db")
df_sql = pd.read_sql("SELECT * FROM customers", conn)
conn.close()

print("Loading API JSON...")
with open("customers_churn_api_sample.json") as f:
    api_data = json.load(f)

df_api = pd.DataFrame(api_data["data"])

print("\nData Loaded Successfully!")

# 3. MERGE DATASETS

df = pd.concat([df_csv, df_json, df_sql, df_api], ignore_index=True)

print("Merged Shape:", df.shape)

# 4. DATA UNDERSTANDING

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())

# 5. DATA CLEANING

# Remove duplicates
df.drop_duplicates(inplace=True)

# Separate columns
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include='object').columns

# Fill missing values
for col in num_cols:
    df[col].fillna(df[col].mean(), inplace=True)

for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Drop ID column
if "CustomerID" in df.columns:
    df.drop("CustomerID", axis=1, inplace=True)

print("\nAfter Cleaning Missing Values:")
print(df.isnull().sum())

# 6. EXPLORATORY DATA ANALYSIS

# Churn Distribution
plt.figure()
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.show()

# Age Distribution
plt.figure()
sns.histplot(df["Age"], kde=True)
plt.title("Age Distribution")
plt.show()

# Income Distribution
plt.figure()
sns.histplot(df["Income"], kde=True)
plt.title("Income Distribution")
plt.show()

# Gender vs Churn
plt.figure()
sns.countplot(x="Gender", hue="Churn", data=df)
plt.title("Gender vs Churn")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 7. FEATURE ENGINEERING

# Create new feature
df["Avg_Spending"] = df["Total_Spending"] / df["Purchase_Frequency"]

# Encode categorical columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# 8. MODEL BUILDING

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("\n===== Logistic Regression =====")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))

# Random Forest
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\n===== Random Forest =====")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Feature Importance
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).plot(kind="bar", figsize=(10,6))
plt.title("Feature Importance")
plt.show()

print("\nPROJECT COMPLETED SUCCESSFULLY ")