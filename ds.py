import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("gender_classification.csv")
print("Summary Statisctics", df.describe())


# pip install pandas numpy matplotlib seaborn scikit-learn


le = LabelEncoder()
# df['gender'] = df['gender'].map({'Male':0, 'Female':1})
# df['gender'] = le.fit_transform(df['gender'])

for column in df.columns:
    if df[column].dtype == 'object':  # Check if the column is categorical (dtype='object')
        df[column] = le.fit_transform(df[column])
        
        df.fillna(df.mode().iloc[0])
    else:
        df.fillna(df.mean(),inplace=True)
print("Missing:\n", df.isnull().sum())
print("Missing:\n", df.isnull().sum())

corrMat = df.corr()
print("Correlation matrix:\n", corrMat)

plt.figure(figsize=(10,8))
sns.heatmap(corrMat, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = df.drop(columns=['gender'])
y = df['gender']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_logreg_pred = logreg.predict(X_test)

print("Accuracy of Log reg: ", accuracy_score(y_test, y_logreg_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_logreg_pred))
print("Classification Report:\n", classification_report(y_test, y_logreg_pred))

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_rf_pred = rf.predict(X_test)

print("Accuracy of Random Forest: ", accuracy_score(y_test, y_rf_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_rf_pred))
print("Classification Report:\n", classification_report(y_test, y_rf_pred))

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_knn_pred = knn.predict(X_test)

print("Accuracy of  KNN: ", accuracy_score(y_test, y_knn_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_knn_pred))
print("Classification Report:\n", classification_report(y_test, y_knn_pred))

# Training accuracy for Logistic Regression
y_train_pred_logreg = logreg.predict(X_train)
print("Training Accuracy of Log reg: ", accuracy_score(y_train, y_train_pred_logreg))
print("Accuracy of Log reg: ", accuracy_score(y_test, y_logreg_pred))

# Training accuracy for Random Forest
y_train_pred_rf = rf.predict(X_train)
print("Training Accuracy of Random Forest: ", accuracy_score(y_train, y_train_pred_rf))
print("Accuracy of Random Forest: ", accuracy_score(y_test, y_rf_pred))

# Training accuracy for KNN
y_train_pred_knn = knn.predict(X_train)
print("Training Accuracy of KNN: ", accuracy_score(y_train, y_train_pred_knn))
print("Accuracy of  KNN: ", accuracy_score(y_test, y_knn_pred))

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

svr = SVR()
svr.fit(X_train, y_train)
y_svr_pred = svr.predict(X_test)

print('Root mean squared error: ', mean_squared_error(y_test, y_svr_pred))
print('r2-score: ', r2_score(y_test, y_svr_pred))

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
y_rfr_pred = rfr.predict(X_test)

print('Root mean squared error: ', mean_squared_error(y_test, y_rfr_pred))
print('r2-score: ', r2_score(y_test, y_rfr_pred))

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
y_lr_pred = lr.predict(X_test)

print('Root mean squared error: ', mean_squared_error(y_test, y_lr_pred))
print('r2-score: ', r2_score(y_test, y_lr_pred))

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_lr_pred, color='blue', label='actual data')
plt.plot([min(y_test), max(y_test)],[min(y_test), max(y_test)], color='red', linestyle='--', label='reg line')
plt.title("Linear Regression")
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.show()