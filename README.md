# Bank Customer Churn: EDA, Dashboarding, Machine Learning and Deep Learning

## Overview

This project aims to predict customer churn for a banking institution using machine learning and deep learning models. The dataset used in this analysis contains customer data related to churn, downloaded from [Kaggle](https://www.kaggle.com/datasets/bhuviranga/customer-churn-data).

The project consists of four main parts:

1- Power BI Dashboard: An interactive dashboard for visualizing customer churn insights

2- Exploratory Data Analysis (EDA) and Data Visualization in Python

3- Model Training: Using machine learning and deep learning to preprocess the dataset and develop predictive models

4- Model Deployment: A web application built with Flask to serve the prediction models

## 
To run the project locally:

1- Clone the project repository: ```git clone <repository-url>``` and ```cd bank_customer_churn_prediction```.

2- Install dependencies: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost, TensorFlow/Keras, Flask, Power BI.

3- Explore data and run Jupyter Notebooks (EDA, Preprocessing, and Model Training): Navigate to the ```Dataset and Notebooks``` folder and run ```jupyter notebook_name.ipynb```.

4- Model Training: Navigate to the ```Model Training``` folder and run ```python model_training.py```. This will create three trained model versions: model_v1.pickle (Random Forest), model_v2.pickle (XGBoost), and model_v3.pickle (Artificial Neural Network).

5- Run the Flask Web Application for Model Deployment: Navigate to the ```Model Deployment``` folder and start the Flask application by running: ```python app.py```.

6- Access the web app: By default, the app runs on ```http://localhost:5000```. Open your web browser and visit this URL.

## Requirements
- Python
- Jupyter Notebooks
- Dependencies: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost, TensorFlow/Keras, Flask, Power BI 




