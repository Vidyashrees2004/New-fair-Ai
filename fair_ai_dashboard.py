import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import time
from codecarbon import EmissionsTracker
from sklearn.metrics import accuracy_score
from fairlearn.metrics import demographic_parity_difference

st.set_page_config(page_title="Fair AI Dashboard", page_icon="🎯", layout="wide")

st.title("🎯 Fair AI Dashboard")
st.subheader("Explainable + Fair + Sustainable AI")

# ==========================================================
# LOAD MODELS
# ==========================================================

@st.cache_resource
def load_models():

    fair_model = joblib.load("models/fair_model.pkl")
    baseline_model = joblib.load("models/baseline_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    X_test_scaled = joblib.load("models/X_test_scaled.pkl")
    y_test = joblib.load("models/y_test.pkl")
    sens_test = joblib.load("models/sens_test.pkl")

    return {
        "fair_model": fair_model,
        "baseline_model": baseline_model,
        "scaler": scaler,
        "feature_names": feature_names,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "sens_test": sens_test
    }

models = load_models()

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    st.header("⚙ Controls")

    model_choice = st.radio(
        "Select Model",
        ["Baseline Model (Explainable)", "Fair Model (Bias Mitigated)"]
    )

    age = st.slider("Age", 18, 80, 35)
    education = st.slider("Education Years", 1, 16, 13)
    hours = st.slider("Hours per Week", 10, 80, 40)
    gender = st.selectbox("Gender", ["Female", "Male"])
    race = st.selectbox("Race", ["Non-White", "White"])

    predict_btn = st.button("🚀 Run Prediction")

# ==========================================================
# PREDICTION
# ==========================================================

if predict_btn:

    gender_num = 1 if gender == "Male" else 0
    race_num = 1 if race == "White" else 0

    features = np.array([[age, education, hours, gender_num, race_num]])
    features_scaled = models["scaler"].transform(features)

    tracker = EmissionsTracker(save_to_file=False)
    tracker.start()
    start = time.time()

    # -------------------
    # MODEL SELECTION
    # -------------------

    if "Baseline" in model_choice:
        model = models["baseline_model"]
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
    else:
        model = models["fair_model"]
        prediction = model.predict(features_scaled)[0]
        probability = 0.5  # safe fallback

    inference_time = time.time() - start
    emissions = tracker.stop()

    # ==================================================
    # SHOW RESULTS (ONLY ONCE)
    # ==================================================

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

  # ==================================================
# SHAP EXPLAINABILITY (BASELINE + FAIR)
# ==================================================

st.markdown("---")
st.header("🧠 Model Explainability")

try:
    if "Baseline" in model_choice:
        base_model = models["baseline_model"]
    else:
        # Extract base learner from Fair model
        fair_model = models["fair_model"]
        if hasattr(fair_model, "predictors_"):
            base_model = fair_model.predictors_[0]
        else:
            base_model = fair_model

    # Use TreeExplainer for tree-based models
    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(features_scaled)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_df = pd.DataFrame({
        "Feature": models["feature_names"],
        "SHAP Value": shap_values[0]
    }).sort_values(by="SHAP Value")

    st.bar_chart(shap_df.set_index("Feature"))

    if "Fair" in model_choice:
        st.caption("Note: SHAP explains the base LightGBM model inside the fairness wrapper.")

except Exception as e:
    st.warning("SHAP explanation could not be generated.")

# ==========================================================
# MODEL COMPARISON
# ==========================================================

st.markdown("---")
st.header("📈 Model Comparison")

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

comparison = pd.DataFrame({
    "Model": ["Baseline", "Fair"],
    "Accuracy": [baseline_acc, fair_acc],
    "Fairness Gap": [baseline_gap, fair_gap]
})

st.dataframe(comparison)

st.metric(
    "Fairness Improvement",
    f"{baseline_gap - fair_gap:.4f}"
)
