

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import root_mean_squared_error



def load_dataset(file_path):
    if not os.path.exists(file_path):
        print("Error: File not found.")
        return None
    df = pd.read_csv(file_path)  
    return df


def explore_dataset(df):
    if df is None:
        print("No dataset to explore.")
        return
    
    print("\nSummary Statistics:")
    print(df.describe())

    print("\nMissing Values:")
    print(df.isnull().sum())

    numeric_df = df.select_dtypes(include=['number'])
    
    print("\nCorrelation Matrix:")
    print(numeric_df.corr())

    if not numeric_df.empty:
        plt.figure(figsize=(10,6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Heatmap")
        plt.show()
    else:
        print("No numeric columns to compute correlations.")


def handle_missing_values(df):
    if df is None:
        print("No dataset loaded. Cannot handle missing values.")
        return None
    
    
    for column in df.columns:
        if df[column].dtype == np.number:
            df[column].fillna(df[column].mean(), inplace=True)  
        else:
            df[column].fillna(df[column].mode()[0], inplace=True)  
    
    return df




file_path = "Icecream.csv" 
df = load_dataset(file_path)
explore_dataset(df)

if df is not None:
    print("\nMissing Values Before Handling:")
    print(df.isnull().sum())
    
    df = handle_missing_values(df)
    
    if df is not None:
        print("\nMissing Values After Handling:")
        print(df.isnull().sum())



label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])


X = df.drop(columns=['Did it rain on that day?'])  
y = df['Did it rain on that day?']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


print("\nSample of Scaled Features:")
print(pd.DataFrame(X_scaled, columns=X.columns).head())
print("Feature scaling success")


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("splitted")

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"\nClassification Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))


accuracy = accuracy_score(y_test, y_pred)

print("Classification Report:\n", classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
#print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

#print("ROC-AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr"))
print("ROC-AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))  # Fix applied


# ---------------- Regression Model ----------------

y_reg = y.astype(float)  

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)

reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train_reg, y_train_reg)
y_pred_reg = reg.predict(X_test_reg)

# Regression Evaluation
print("\n----- Regression Model Performance -----")
print("Mean Squared Error (MSE):", mean_squared_error(y_test_reg, y_pred_reg))
print("Root Mean Squared Error (RMSE):", root_mean_squared_error(y_test_reg, y_pred_reg))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test_reg, y_pred_reg))
print("R² Score:", r2_score(y_test_reg, y_pred_reg))