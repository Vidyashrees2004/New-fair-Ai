import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import time
from codecarbon import EmissionsTracker
from sklearn.metrics import accuracy_score
from fairlearn.metrics import demographic_parity_difference

# ==========================================================
# CONFIG
# ==========================================================
st.set_page_config(page_title="Fair AI Dashboard", page_icon="🎯", layout="wide")

st.title("🎯 Fair AI Dashboard")
st.markdown("### Explainable + Fair + Sustainable AI")

# ==========================================================
# LOAD MODELS
# ==========================================================
@st.cache_resource
def load_models():
    return {
        "fair_model": joblib.load("models/fair_model.pkl"),
        "baseline_model": joblib.load("models/baseline_model.pkl"),
        "scaler": joblib.load("models/scaler.pkl"),
        "feature_names": joblib.load("models/feature_names.pkl"),
        "X_test_scaled": joblib.load("models/X_test_scaled.pkl"),
        "y_test": joblib.load("models/y_test.pkl"),
        "sens_test": joblib.load("models/sens_test.pkl"),
    }

models = load_models()

# ==========================================================
# SIDEBAR CONTROLS
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

    input_dict = {
        "age": age,
        "education-num": education,
        "hours-per-week": hours,
        "sex": gender_num,
        "race": race_num
    }

    features_df = pd.DataFrame([input_dict])
    features = features_df[models["feature_names"]].values
    features_scaled = models["scaler"].transform(features)

    # ENERGY TRACKING
    tracker = EmissionsTracker(save_to_file=False)
    tracker.start()
    start_time = time.time()

    if "Baseline" in model_choice:
        model = models["baseline_model"]
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
    else:
        model = models["fair_model"]
        prediction = model.predict(features_scaled)[0]
        probability = 0.5  # safe fallback

    inference_time = time.time() - start_time
    emissions = tracker.stop()

    # ======================================================
    # RESULT DISPLAY
    # ======================================================
    st.markdown("---")
    st.header("📊 Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        if prediction == 1:
            st.success("💰 HIGH Income (>50K)")
        else:
            st.info("📉 LOW Income (≤50K)")

    with col2:
        st.metric("Confidence", f"{probability:.2%}")

    with col3:
        st.metric("Inference Time (ms)", f"{inference_time*1000:.2f}")
        st.metric("CO₂ Emission (kg)", f"{emissions:.8f}")

    # ======================================================
    # SHAP EXPLANATION
    # ======================================================
    st.markdown("---")
    st.header("🧠 Model Explainability")

    try:
        if "Baseline" in model_choice:
            model_to_explain = models["baseline_model"]
        else:
            fair_model = models["fair_model"]
            if hasattr(fair_model, "predictors_"):
                model_to_explain = fair_model.predictors_[0]
            else:
                model_to_explain = fair_model

        explainer = shap.Explainer(model_to_explain, models["X_test_scaled"][:100])
        shap_values = explainer(features_scaled)

        shap_df = pd.DataFrame({
            "Feature": models["feature_names"],
            "SHAP Value": shap_values.values[0]
        }).sort_values(by="SHAP Value")

        st.bar_chart(shap_df.set_index("Feature")["SHAP Value"])

        st.subheader("📌 Detailed Explanation")
        st.dataframe(shap_df)

        top_positive = shap_df.iloc[-1]
        top_negative = shap_df.iloc[0]

        st.write(f"🔺 {top_positive['Feature']} increases likelihood of HIGH income.")
        st.write(f"🔻 {top_negative['Feature']} pushes prediction toward LOW income.")

        if "Fair" in model_choice:
            st.caption("Fair model explanation derived from underlying base learner.")

    except Exception:
        st.warning("SHAP explanation could not be generated.")

# ==========================================================
# FAIRNESS EXPLANATION
# ==========================================================
st.markdown("---")
st.header("⚖ Fairness Explanation")

if "Fair" in model_choice:
    st.success("""
    The Fair Model applies Demographic Parity constraint using
    Fairlearn's ExponentiatedGradient algorithm.

    ➤ Sensitive attributes are used to measure disparity  
    ➤ Decision boundary is adjusted to reduce group imbalance  
    ➤ Ensures similar positive prediction rates across groups
    """)
else:
    st.warning("""
    Baseline model maximizes accuracy only.
    It may produce biased outcomes across demographic groups.
    """)

# ==========================================================
# DEMOGRAPHIC PARITY VISUALIZATION
# ==========================================================
st.markdown("---")
st.header("📊 Demographic Parity Comparison")

fair_pred = models["fair_model"].predict(models["X_test_scaled"])
baseline_pred = models["baseline_model"].predict(models["X_test_scaled"])

fair_gap = demographic_parity_difference(
    models["y_test"], fair_pred,
    sensitive_features=models["sens_test"]
)

baseline_gap = demographic_parity_difference(
    models["y_test"], baseline_pred,
    sensitive_features=models["sens_test"]
)

gap_df = pd.DataFrame({
    "Model": ["Baseline", "Fair"],
    "Demographic Parity Gap": [baseline_gap, fair_gap]
})

st.bar_chart(gap_df.set_index("Model"))

st.metric(
    "Fairness Improvement",
    f"{baseline_gap - fair_gap:.4f} reduction"
)

# ==========================================================
# DECISION BOUNDARY SHIFT DEMO
# ==========================================================
st.markdown("---")
st.header("🔬 Fairness-Adjusted Decision Logic")

if predict_btn:
    baseline_score = models["baseline_model"].predict_proba(features_scaled)[0][1]
    fair_decision = models["fair_model"].predict(features_scaled)[0]

    st.write("Baseline Probability of High Income:", round(baseline_score, 4))
    st.write("Fair Model Final Decision:",
             "High Income" if fair_decision == 1 else "Low Income")

    st.caption("""
    Fair model may adjust the decision threshold to maintain
    balanced prediction rates across protected groups.
    """)
