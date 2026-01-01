import joblib
import pandas as pd

from .config import (
    PREPROCESSOR_PATH,
    XGB_MODEL_PATH,
    NUMERIC_COLS,
    CATEGORICAL_COLS,
)


def main():
    """
    Simple example showing how to load the trained model
    and generate a failure probability for a single machine.
    """
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(XGB_MODEL_PATH)

    sample_input = {
        "air_temperature_k": 298.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500,
        "torque_nm": 40.0,
        "tool_wear_min": 120,
        "product_id": "M14860",
        "type": "M",
    }

    input_df = pd.DataFrame([sample_input])
    input_df = input_df[NUMERIC_COLS + CATEGORICAL_COLS]

    processed_input = preprocessor.transform(input_df)
    failure_probability = model.predict_proba(processed_input)[0, 1]

    print(f"Predicted failure probability: {failure_probability:.3f}")


if __name__ == "__main__":
    main()
