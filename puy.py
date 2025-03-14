
# Step 1: Load and Explore the Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (Replace with your actual filename)
df = pd.read_csv('diabetes_main.csv')

# Display first few rows
print("Dataset Preview:")
print(df.head())

# Summary statistics (Descriptive Statistics)
print("\nDataset Summary:")
print(df.describe().T) # Transposed for better readability

# Ensure only numeric columns are used
df_numeric = df.select_dtypes(include=[np.number])

# Compute descriptive statistics safely
print("Mean Values:\n", df_numeric.mean())
print("Median Values:\n", df_numeric.median())
print("Mode Values:\n", df_numeric.mode().iloc[0])
print("Standard Deviation:\n", df_numeric.std())
print("Variance:\n", df_numeric.var())
print("Skewness:\n", df_numeric.skew())
print("Kurtosis:\n", df_numeric.kurt())


# Step 2: Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Handle missing values (fill with mean for numerical & mode for categorical)
for col in df.columns:
    if df[col].dtype == 'object': # Categorical feature
        df[col] = df[col].fillna(df[col].mode()[0])
    else: # Numerical feature
        df[col] = df[col].fillna(df[col].mean()) # Fixed inplace warning

# Compute and display the Correlation Matrix using only numeric columns
df_numeric = df.select_dtypes(include=[np.number]) # Select only numeric columns
print("\nCorrelation Matrix:")
correlation_matrix = df_numeric.corr()
print(correlation_matrix)

# Visualizing Correlation Matrix
plt.figure(figsize=(10,6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Compute and display the Covariance Matrix using only numeric columns
df_numeric = df.select_dtypes(include=[np.number]) # Select only numeric columns
print("\nCovariance Matrix:")
covariance_matrix = df_numeric.cov()
print(covariance_matrix)

# Define features (X) and target (y) - Adjust based on dataset
target_column = 'Outcome' # Change this if needed
if target_column not in df.columns:
    raise ValueError(f"Column '{target_column}' not found in dataset!")

X = df.drop(columns=[target_column])
y = df[target_column]

# Encode categorical features using LabelEncoder
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col]) # Convert text to numeric labels
    label_encoders[col] = le # Store encoder for inverse transformation if needed

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Standardization) - Now only numeric features are present
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Step 3: Train a Decision Tree Model
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# Step 4: Model Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predictions
y_pred = model.predict(X_test)

# Performance Metrics
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Visualization
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

print("The Decision Tree model achieved an accuracy of", round(accuracy_score(y_test, y_pred) * 100, 2), "%.")




# Below code for Prediction

'''
# Step 3: Train a Decision Tree Model
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(criterion='squared_error', random_state=42)
model.fit(X_train, y_train)

# Step 4: Model Evaluation (Regression Metrics)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Predictions
y_pred = model.predict(X_test)

# Performance Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared Score (R²):", r2)

print(f"The Decision Tree model achieved an R-squared score of {round(r2 * 100, 2)}%.")'