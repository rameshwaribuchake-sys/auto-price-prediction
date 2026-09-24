# 🚗 Auto Price Prediction Dashboard

A Machine Learning-based web dashboard that predicts automobile prices using **Decision Tree Regression** and **Random Forest Regression**. The project provides an interactive interface built with **Streamlit** where users can enter vehicle details and obtain a predicted price.

## 📌 Project Overview

The **Auto Price Prediction Dashboard** uses automobile data to train regression models and predict the price of a vehicle based on its characteristics.

Two Machine Learning algorithms are used:

* Decision Tree Regressor
* Random Forest Regressor

The trained models are saved as `.pkl` files and integrated into a Streamlit dashboard for interactive predictions.

## 🎯 Objectives

* Predict automobile prices using Machine Learning.
* Compare Decision Tree and Random Forest regression models.
* Build an interactive prediction dashboard using Streamlit.
* Understand the relationship between automobile features and price.
* Provide a simple interface for making vehicle price predictions.

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Streamlit**
* **Pickle**
* **Jupyter Notebook**

## 🤖 Machine Learning Models

### 1. Decision Tree Regressor

A Decision Tree Regressor predicts the target value by splitting the dataset into different decision-based branches.

### 2. Random Forest Regressor

Random Forest combines multiple decision trees to produce a more robust regression prediction.

The dashboard allows users to use the trained models for automobile price prediction.

## 📊 Dashboard Features

* Interactive automobile feature input
* Automobile price prediction
* Decision Tree prediction
* Random Forest prediction
* Model performance comparison
* Feature importance visualization
* Simple and user-friendly Streamlit interface

## 📁 Project Structure

```text
auto-price-prediction-dashboard/
│
├── app.py
├── dt_reg.pkl
├── rf_model.pkl
├── feature_columns.pkl
├── preprocess_info.pkl
├── processed_auto.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/auto-price-prediction-dashboard.git
```

### 2. Open the project folder

```bash
cd auto-price-prediction-dashboard
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

## ▶️ Run the Dashboard

Run the following command:

```bash
streamlit run app.py
```

The Streamlit dashboard will open in your web browser.

## 📈 Model Evaluation

The project evaluates the regression models using common regression metrics:

* **R² Score**
* **Mean Absolute Error (MAE)**
* **Root Mean Squared Error (RMSE)**

These metrics are used to evaluate and compare model performance.

## 🔄 Workflow

```text
Automobile Dataset
        ↓
Data Preprocessing
        ↓
Feature Selection
        ↓
Train-Test Split
        ↓
Model Training
   ↙            ↘
Decision Tree   Random Forest
   ↓               ↓
Model Evaluation
        ↓
Save Models as .pkl
        ↓
Streamlit Dashboard
        ↓
Automobile Price Prediction
```

## 💡 Future Improvements

* Add more Machine Learning algorithms.
* Improve dashboard visualization.
* Add prediction history.
* Deploy the dashboard online.
* Add additional automobile datasets.
* Improve model accuracy through hyperparameter tuning.

## 👩‍💻 Author

**Rameshwari Buchake**

B.E. Artificial Intelligence & Data Science

## ⭐ Acknowledgement

This project was developed as a Machine Learning and Streamlit dashboard project to demonstrate automobile price prediction using regression techniques.
