"""Universal Prediction System - Streamlit web app.
Loads any trained pipeline and adapts its UI to the model's requirements."""

import json
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Universal Prediction System", page_icon="🔮", layout="centered")

@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_info():
    with open("model_info.json") as f:
        return json.load(f)

try:
    saved_data = load_model()
    info = load_info()
except FileNotFoundError:
    st.error("⚠️ Model files not found. Please ensure `best_model.pkl` and `model_info.json` are in the same folder.")
    st.stop()

model = saved_data["model"]
features = saved_data["features"]
task = saved_data["task"]
target = saved_data["target"]

st.title(f"🔮 {target.capitalize()} Prediction System")
st.caption(
    f"Model: **{info['best_algorithm']}** · trained on {info['n_rows']:,} rows · "
    f"Task: {task.capitalize()}"
)

st.subheader("Enter the details")

# Create a dictionary to hold the user's input
input_data = {}

# Use columns to make the form look nice
col1, col2 = st.columns(2)

for i, feature in enumerate(features):
    # Get the metadata for this specific feature from the JSON
    meta = info["features_meta"][feature]
    dtype = meta["dtype"]
    
    # Alternate between column 1 and column 2
    container = col1 if i % 2 == 0 else col2
    
    with container:
        if dtype == "numeric":
            # If it's numeric, use a number input with the median as default
            default_val = meta.get("median", 0.0)
            # Use a slider if the range is reasonable, otherwise number input
            input_data[feature] = st.number_input(
                label=feature.replace("_", " ").title(),
                value=float(default_val),
                help=f"Typical value: {default_val}"
            )
        else:
            # If it's categorical, use a selectbox with the most common values
            options = meta.get("top_values", [])
            input_data[feature] = st.selectbox(
                label=feature.replace("_", " ").title(),
                options=options
            )

if st.button("Predict", type="primary"):
    # Convert input dictionary to a DataFrame
    input_df = pd.DataFrame([input_data])
    
    try:
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Display result based on the task
        if task == "regression":
            # For regression, show the number and a typical error range
            mae = info.get("cv_mean_mae", 0)
            st.success(f"### Predicted {target.capitalize()}: {prediction:,.2f}")
            if mae > 0:
                st.write(
                    f"Typical error (MAE): about **±{mae:,.2f}**, "
                    f"so a realistic range is **{max(prediction - mae, 0):,.2f} – {prediction + mae:,.2f}**."
                )
        else:
            # For classification, just show the class
            st.success(f"### Predicted {target.capitalize()}: **{prediction}**")
            
        with st.expander("Show the feature vector sent to the model"):
            st.dataframe(input_df)
            
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.caption("Universal Prediction System. Built with Streamlit and scikit-learn.")