import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def train_baseline_model(X, y):
    """
    Train a simple baseline model for comparison.
    Logistic Regression works well as a first benchmark.
    """
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X, y)
    return model


def train_xgboost_model(X, y):
    """
    Train the main XGBoost classifier.
    This model handles non-linear relationships and class imbalance better.
    """
    positive_count = (y == 1).sum()
    negative_count = (y == 0).sum()

    scale_pos_weight = (
        negative_count / positive_count if positive_count > 0 else 1.0
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        n_jobs=-1,
    )

    model.fit(X, y)
    return model
