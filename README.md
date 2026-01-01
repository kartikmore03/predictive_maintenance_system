# Predictive Maintenance for Industrial Equipment (AI4I 2020)

This project focuses on predicting **machine failures** using historical operational and sensor data from industrial equipment.  
The main idea is to support **predictive maintenance** by estimating the probability of a machine failing ahead of time, allowing maintenance to be scheduled before a breakdown actually happens.

---

## Dataset

- **AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning Repository)  
- Download the dataset (usually named `ai4i2020.csv`) and place it inside the `data/` directory.

The dataset includes information such as air and process temperature, rotational speed, torque, tool wear, and a binary label indicating whether a machine failure occurred.

---

## Tech Stack

- Python  
- Pandas and NumPy  
- Scikit-learn  
- XGBoost  
- Joblib  

---

## Project Structure

- **`src/data_utils.py`**  
  Handles data loading, column renaming, preprocessing, and train–test splitting.

- **`src/models.py`**  
  Contains the baseline Logistic Regression model and the main XGBoost model used for training.

- **`src/train.py`**  
  Trains both models, evaluates their performance, and saves the trained models along with the preprocessing pipeline.

- **`src/predict_example.py`**  
  Shows how to load the saved model and preprocessor to predict the failure probability for a single machine input.

---

## How to Run

```bash
pip install -r requirements.txt

# Train the models
python -m src.train

# Run a sample prediction
python -m src.predict_example
