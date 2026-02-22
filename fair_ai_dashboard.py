import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import shap
import joblib
import time
import psutil
from codecarbon import EmissionsTracker

st.set_page_config(page_title="Fair AI Dashboard", page_icon="🤖", layout="wide")

st.title("🎯 Fair AI Dashboard")
st.subheader("Model Explainability + Energy Tracking + Fairness Metrics")

# ==========================================================
# LOAD MODELS
# ==========================================================

@st.cache_resource
def load_models():
    try:
        fair_model = joblib.load("models/fair_model.pkl")
        baseline_model = joblib.load("models/baseline_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        feature_names = joblib.load("models/feature_names.pkl")
        X_test_scaled = joblib.load("models/X_test_scaled.pkl")
        y_test = joblib.load("models/y_test.pkl")
        sens_test = joblib.load("models/sens_test.pkl")

        # Extract LightGBM from Fair model
        if hasattr(fair_model, "predictors_"):
            fair_base = fair_model.predictors_[0]
        else:
            fair_base = fair_model

        fair_explainer = shap.TreeExplainer(fair_base)
        baseline_explainer = shap.TreeExplainer(baseline_model)

        return {
            "fair_model": fair_model,
            "baseline_model": baseline_model,
            "scaler": scaler,
            "feature_names": feature_names,
            "X_test_scaled": X_test_scaled,
            "y_test": y_test,
            "sens_test": sens_test,
            "fair_explainer": fair_explainer,
            "baseline_explainer": baseline_explainer
        }

    except Exception as e:
        st.error(f"Model Loading Error: {e}")
        return None


models = load_models()
if models is None:
    st.stop()

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    st.header("⚙️ Controls")

    model_choice = st.radio(
        "Select Model",
        ["Fair Model", "Baseline Model"]
    )

    age = st.slider("Age", 18, 80, 35)
    education = st.slider("Education Years", 1, 16, 13)
    hours = st.slider("Hours/Week", 10, 80, 40)
    gender = st.selectbox("Gender", ["Female", "Male"])
    race = st.selectbox("Race", ["Non-White", "White"])

    gender_num = 1 if gender == "Male" else 0
    race_num = 1 if race == "White" else 0

    predict_btn = st.button("🚀 Run Prediction", use_container_width=True)

# ==========================================================
# PREDICTION
# ==========================================================

if predict_btn:

    features = np.array([[age, education, hours, gender_num, race_num]])
    features_scaled = models["scaler"].transform(features)

    tracker = EmissionsTracker(save_to_file=False)
    tracker.start()
    start_time = time.time()

    # ---------------- SAFE MODEL HANDLING ---------------- #

    if model_choice == "Fair Model":
        model = models["fair_model"]
        prediction = model.predict(features_scaled)[0]

        try:
            probability = model.predict_proba(features_scaled)[0][1]
        except:
            # fallback if predict_proba not supported
            score = model.decision_function(features_scaled)
            probability = float(1 / (1 + np.exp(-score)))

        explainer = models["fair_explainer"]

    else:
        model = models["baseline_model"]
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        explainer = models["baseline_explainer"]

    inference_time = time.time() - start_time
    emissions = tracker.stop()

    # ======================================================
    # RESULTS
    # ======================================================

    st.markdown("---")
    st.header("📊 Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        if prediction == 1:
            st.success("💰 HIGH Income (>50K)")
        else:
            st.info("📉 LOW Income (<=50K)")

    with col2:
        st.metric("Confidence", f"{probability:.2%}")

    with col3:
        st.metric("Inference Time (ms)", f"{inference_time*1000:.2f}")
        st.metric("CO₂ Emission (kg)", f"{emissions:.8f}")

    # ======================================================
    # SHAP (SAFE)
    # ======================================================

    st.markdown("---")
    st.header("🧠 SHAP Explainability")

    try:
        shap_values = explainer.shap_values(features_scaled)

        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        shap_df = pd.DataFrame({
            "Feature": models["feature_names"],
            "SHAP Value": shap_values[0]
        })

        shap_df["Abs"] = np.abs(shap_df["SHAP Value"])
        shap_df = shap_df.sort_values("Abs", ascending=True)

        colors = ["#00cc96" if x > 0 else "#EF553B" for x in shap_df["SHAP Value"]]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=shap_df["Feature"],
            x=shap_df["SHAP Value"],
            orientation="h",
            marker_color=colors
        ))

        fig.update_layout(
            height=400,
            title="Feature Contribution to Prediction",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception:
        st.warning("SHAP explanation not available for this input.")

# ==========================================================
# MODEL COMPARISON
# ==========================================================

st.markdown("---")
st.header("📈 Model Performance Comparison")

from sklearn.metrics import accuracy_score
from fairlearn.metrics import demographic_parity_difference

fair_pred = models["fair_model"].predict(models["X_test_scaled"])
baseline_pred = models["baseline_model"].predict(models["X_test_scaled"])

fair_acc = accuracy_score(models["y_test"], fair_pred)
baseline_acc = accuracy_score(models["y_test"], baseline_pred)

fair_gap = demographic_parity_difference(
    models["y_test"], fair_pred, sensitive_features=models["sens_test"]
)
baseline_gap = demographic_parity_difference(
    models["y_test"], baseline_pred, sensitive_features=models["sens_test"]
)

comparison_df = pd.DataFrame({
    "Model": ["Fair Model", "Baseline"],
    "Accuracy": [fair_acc, baseline_acc],
    "Fairness Gap": [fair_gap, baseline_gap]
})

st.dataframe(comparison_df)

st.metric(
    "Fairness Improvement",
    f"{baseline_gap - fair_gap:.4f}"
)
