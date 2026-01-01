import os
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from .config import (
    MODELS_DIR,
    PREPROCESSOR_PATH,
    BASELINE_MODEL_PATH,
    XGB_MODEL_PATH,
)
from .data_utils import load_data, rename_columns, prepare_data
from .models import train_baseline_model, train_xgboost_model


def evaluate_model(name, model, X_test, y_test):
    """
    Print standard classification metrics for a trained model.
    """
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, probabilities):.4f}")
    print(classification_report(y_test, predictions, digits=4))


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading dataset...")
    df = load_data()
    df = rename_columns(df)

    print("Preparing train and test data...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    print("Training baseline model...")
    baseline_model = train_baseline_model(X_train, y_train)
    joblib.dump(baseline_model, BASELINE_MODEL_PATH)
    evaluate_model("Baseline Logistic Regression", baseline_model, X_test, y_test)

    print("Training XGBoost model...")
    xgb_model = train_xgboost_model(X_train, y_train)
    joblib.dump(xgb_model, XGB_MODEL_PATH)
    evaluate_model("XGBoost Model", xgb_model, X_test, y_test)


if __name__ == "__main__":
    main()
