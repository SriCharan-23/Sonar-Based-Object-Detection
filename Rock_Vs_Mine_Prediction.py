# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pickle

# Reading data
sonar_data = pd.read_csv("Copy of sonar data.csv", header=None)

# Splitting features and target
x = sonar_data.drop(columns=60, axis=1)
y = sonar_data[60]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, stratify=y, random_state=1
)

# Creating models with preprocessing (scaling included)
lr = make_pipeline(StandardScaler(), LogisticRegression())
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
svm = make_pipeline(StandardScaler(), SVC(kernel="linear"))

# Training models
lr.fit(x_train, y_train)
knn.fit(x_train, y_train)
svm.fit(x_train, y_train)

# Predictions
y_lrPredict = lr.predict(x_test)
y_knnPredict = knn.predict(x_test)
y_svmPredict = svm.predict(x_test)

# Accuracy
acc_lr = accuracy_score(y_test, y_lrPredict)
acc_knn = accuracy_score(y_test, y_knnPredict)
acc_svm = accuracy_score(y_test, y_svmPredict)

# Store models
models = {
    "Logistic Regression": lr,
    "KNN": knn,
    "SVM": svm
}

accuracies = {
    "Logistic Regression": acc_lr,
    "KNN": acc_knn,
    "SVM": acc_svm
}

# Save models
pickle.dump(models, open('model.pkl', 'wb'))
pickle.dump(accuracies, open('accuracy.pkl', 'wb'))






