import os
import sys

import joblib
import pandas as pd
import streamlit as st

# Ensure the src/ directory is available for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from config import (
    PREPROCESSOR_PATH,
    XGB_MODEL_PATH,
    NUMERIC_COLS,
    CATEGORICAL_COLS,
)


@st.cache_resource
def load_artifacts():
    """
    Load the trained model and preprocessing pipeline.
    Cached so they are loaded only once per session.
    """
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(XGB_MODEL_PATH)
    return preprocessor, model


def main():
    st.set_page_config(
        page_title="Predictive Maintenance – Failure Risk",
        layout="centered",
    )

    st.title("🛠️ Predictive Maintenance Dashboard")
    st.write(
        """
        This dashboard estimates the **failure risk** of an industrial machine
        using a trained machine learning model (XGBoost) 
        """
    )

    # Load model and preprocessor
    try:
        preprocessor, model = load_artifacts()
    except Exception as error:
        st.error(
            "Unable to load the model or preprocessing pipeline. "
            "Please make sure the training script has been run first."
        )
        st.exception(error)
        return

    st.subheader("Machine Operating Conditions")

    # Numeric inputs with realistic ranges
    air_temp = st.slider(
        "Air temperature [K]",
        min_value=290.0,
        max_value=320.0,
        value=298.0,
        step=0.1,
    )
    process_temp = st.slider(
        "Process temperature [K]",
        min_value=300.0,
        max_value=340.0,
        value=310.0,
        step=0.1,
    )
    rotational_speed = st.slider(
        "Rotational speed [rpm]",
        min_value=500,
        max_value=3000,
        value=1500,
        step=10,
    )
    torque = st.slider(
        "Torque [Nm]",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=0.5,
    )
    tool_wear = st.slider(
        "Tool wear [min]",
        min_value=0,
        max_value=250,
        value=100,
        step=1,
    )

    st.subheader("Machine / Product Details")

    # Categorical inputs
    # product_id is free text; unseen values are handled safely by the encoder
    product_id = st.text_input("Product ID", value="M14860")

    # Typical product types in the AI4I dataset
    product_type = st.selectbox(
        "Product type",
        options=["L", "M", "H"],
        index=1,
    )

    # Build a single-row dataframe for prediction
    input_data = {
        "air_temperature_k": air_temp,
        "process_temperature_k": process_temp,
        "rotational_speed_rpm": rotational_speed,
        "torque_nm": torque,
        "tool_wear_min": tool_wear,
        "product_id": product_id,
        "type": product_type,
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df[NUMERIC_COLS + CATEGORICAL_COLS]

    st.markdown("---")

    if st.button("Predict Failure Risk"):
        processed_input = preprocessor.transform(input_df)
        failure_probability = model.predict_proba(processed_input)[0, 1]

        st.write("### Prediction Result")
        st.write(
            f"**Estimated failure probability:** `{failure_probability:.3f}`"
        )

        if failure_probability >= 0.5:
            st.error(
                "❗ High risk of failure detected. "
                "Recommended action: schedule maintenance soon."
            )
        else:
            st.success(
                "✅ Low risk of immediate failure based on current inputs."
            )

        st.caption(
            "Note: The probability threshold is set to 0.5 for demonstration. "
            "In a production system, this would be adjusted based on business needs."
        )




if __name__ == "__main__":
    main()
