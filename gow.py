# Step 1: Load and Explore the Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib

# Ensure plots display correctly
matplotlib.use('TkAgg')

# Load dataset
df = pd.read_csv(r'C:\LAB\ds\Iris.csv')

# Display dataset preview
print("🔹 Dataset Preview:\n", df.head())

# Summary statistics
print("\n🔹 Dataset Summary:\n", df.describe().T)

# Check for missing values
missing_values = df.isnull().sum()
print("\n🔹 Missing Values:\n", missing_values)

# Handle missing values (fill with mean for numerical & mode for categorical)
for col in df.columns:
    if df[col].dtype == 'object':  # Categorical feature
        df[col] = df[col].fillna(df[col].mode()[0])  # Fixed FutureWarning
    else:  # Numerical feature
        df[col] = df[col].fillna(df[col].mean())  # Fixed FutureWarning

# Compute and display correlation matrix
df_numeric = df.select_dtypes(include=[np.number])  # Only numeric columns
correlation_matrix = df_numeric.corr()
print("\n🔹 Correlation Matrix:\n", correlation_matrix)

# Visualizing Correlation Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show(block=True)  # Ensures plot is displayed

# Compute and display covariance matrix
print("\n🔹 Covariance Matrix:\n", df_numeric.cov())

# Define features (X) and target (y)
target_column = 'Species'  # Change this if needed
if target_column not in df.columns:
    raise ValueError(f"❌ Column '{target_column}' not found in dataset!")

X = df.drop(columns=[target_column])
y = df[target_column]

# Encode categorical features (if any)
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])  # Convert text to numeric labels
    label_encoders[col] = le  # Store encoder for inverse transformation if needed

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Train a Decision Tree Model
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# Step 4: Model Evaluation
# Predictions
y_pred = model.predict(X_test)

# Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
print("\n🔹 Model Accuracy:", round(accuracy * 100, 2), "%")
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Visualization
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show(block=True)  # Ensures plot is displayed