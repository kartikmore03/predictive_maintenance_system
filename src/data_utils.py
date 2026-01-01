import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from .config import (
    RAW_DATA_PATH,
    TARGET_COL,
    NUMERIC_COLS,
    CATEGORICAL_COLS,
)


def load_data():
    """
    Load the raw dataset from disk.
    """
    return pd.read_csv(RAW_DATA_PATH)


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
 
    column_mapping = {
        "UDI": "uid",
        "Product ID": "product_id",
        "Type": "type",
        "Air temperature [K]": "air_temperature_k",
        "Process temperature [K]": "process_temperature_k",
        "Rotational speed [rpm]": "rotational_speed_rpm",
        "Torque [Nm]": "torque_nm",
        "Tool wear [min]": "tool_wear_min",
        "Machine failure": "machine_failure",
    }

    df = df.rename(columns={c: column_mapping.get(c, c) for c in df.columns})
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    return df


def build_preprocessor():
    """
    Create preprocessing steps for numeric and categorical features.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ]
    )

    return preprocessor


def prepare_data(df, test_size=0.2, random_state=42):
    """
    Split the data into train and test sets and apply preprocessing.
    """
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor
