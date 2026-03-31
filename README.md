🚀 Rock vs Mine Prediction using Machine Learning

📌 Project Overview

This project uses machine learning to predict whether an object detected by sonar is a Rock or a Mine. 
The system is built with Python and deployed via a Flask web application, allowing users to input sonar data and get real-time predictions.

🎯 Objective

The main goal of this project is to:

Build a classification model to distinguish between rocks and mines
Analyze sonar data patterns
Achieve high prediction accuracy using ML techniques


📊 Dataset Information

The dataset used is the Sonar Dataset
It contains 60 numerical features representing sonar signals
Each instance is labeled as:
R → Rock
M → Mine

🛠️ Technologies Used

Python 
NumPy
Pandas
Scikit-learn
flask
VS Code

⚙️ Workflow

Data Collection
Data Preprocessing
Exploratory Data Analysis (EDA)
Splitting data into training and testing sets
Model Training and scaling
Model Evaluation
Prediction

🤖 Model Used

Logistic Regression
KNN classifier
SVM (linear kernel)

📈 Model Performance

| Model                  | Accuracy |
|------------------------|----------|
| Logistic Regression    | 76.19%   |
| K-Nearest Neighbors    | 90.48%   |
| Support Vector Machine | 76.19%   |


📂 Project Structure

rock-vs-mine-project/
│
├──templates/index.html
├── Copy od sonar data.csv
├── model.pkl
├── accuracy.pkl
├── Rock_Vs_Mine_Prediction.py
├── app.py
├── requirements.txt
└── README.md

The Sonar dataset for this project is taken from "Kaggle".
