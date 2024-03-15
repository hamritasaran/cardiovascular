import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Load the dataset (replace 'heart_disease.csv' with your dataset file)
data = pd.read_csv('/content/drive/MyDrive/Cardiovascular Disease.csv',sep=',')
# Explore the dataset
print(data.head(10))
print(data.info())
print(data.describe())
# Step 1: Data preprocessing
# Convert relevant columns to numeric data types
numeric_columns = ['id', 'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric)
# Drop any rows with missing or NaN values
data.dropna(inplace=True)
# Step 2: Data analysis and visualizations
# Visualization: Distribution of the Target Variable
plt.figure(figsize=(6, 4))
sns.countplot(x='cardio', data=data)
plt.title("Distribution of Target Variable")
plt.show()
# Visualization: Pairplot of Numerical Features colored by Target Variable
sns.pairplot(data, hue='cardio', vars=numeric_columns[1:], diag_kind='kde')
plt.suptitle("Pairplot of Numerical Features colored by Target Variable")
plt.show()
# Create a single boxplot for each numerical feature by the target variable
plt.figure(figsize=(12, 8))
for idx, feature in enumerate(numeric_columns[1:-1]):
 plt.subplot(3, 4, idx+1)
 sns.boxplot(x='cardio', y=feature, data=data)
 plt.title(f"Boxplot of {feature} by Target Variable")
plt.tight_layout()
plt.show()
plt.figure(figsize=(12, 10))
for idx, feature in enumerate(numeric_columns[1:-1]):
 plt.subplot(3, 4, idx+1)
 sns.histplot(data[feature], kde=True)
 plt.title(f"Histogram of {feature}")
plt.tight_layout()
plt.show()
# Step 3: Calculate the correlation matrix
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix Heatmap")
plt.show()
# Step 4: Initialize and train the machine learning models
X = data[numeric_columns[1:-1]] # Features
y = data['cardio'] # Target variable
# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
# Step 4: Initialize and train the machine learning models
svm_model = SVC(kernel='linear')
knn_model = KNeighborsClassifier(n_neighbors=5)
dt_model = DecisionTreeClassifier()
lr_model = LogisticRegression()
rf_model = RandomForestClassifier(n_estimators=100)
svm_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
# Step 5: Make predictions and evaluate accuracy
svm_pred = svm_model.predict(X_test)
knn_pred = knn_model.predict(X_test)
dt_pred = dt_model.predict(X_test)
lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
knn_accuracy = accuracy_score(y_test, knn_pred)
dt_accuracy = accuracy_score(y_test, dt_pred)
lr_accuracy = accuracy_score(y_test, lr_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)
print("Accuracy of SVM:", svm_accuracy)
print("Accuracy of KNN:", knn_accuracy)
print("Accuracy of Decision Trees:", dt_accuracy)
print("Accuracy of Logistic Regression:", lr_accuracy)
print("Accuracy of Random Forest:", rf_accuracy)
# Step 5: Initialize and train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
# Make predictions and evaluate the model
svm_pred = svm_model.predict(X_test)
# Calculate accuracy
svm_accuracy = accuracy_score(y_test, svm_pred)
print("Accuracy of SVM:", svm_accuracy)
# Other evaluation metrics
print("Classification Report:")
print(classification_report(y_test, svm_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, svm_pred))
