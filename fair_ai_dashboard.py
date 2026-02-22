import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
import joblib
import time
import psutil
import os
from codecarbon import EmissionsTracker
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Fair AI Dashboard",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .fairness-good {
        color: #00ff00;
        font-weight: bold;
    }
    .fairness-bad {
        color: #ff4444;
        font-weight: bold;
    }
    .explanation-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1E88E5;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>🎯 Fair AI Dashboard</h1>", unsafe_allow_html=True)
st.markdown("### Model Explainability + Energy Tracking + Fairness Metrics")

# Load models and data
@st.cache_resource
def load_models():
    """Load all models and required data"""
    try:
        fair_model = joblib.load('models/fair_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        X_test_scaled = joblib.load('models/X_test_scaled.pkl')
        y_test = joblib.load('models/y_test.pkl')
        sens_test = joblib.load('models/sens_test.pkl')
        
        # Load baseline for comparison
        baseline_model = joblib.load('models/baseline_model.pkl')
        
        # Create SHAP explainer for fair model
        explainer = shap.TreeExplainer(fair_model._model if hasattr(fair_model, '_model') else fair_model)
        
        return {
            'fair_model': fair_model,
            'baseline_model': baseline_model,
            'scaler': scaler,
            'feature_names': feature_names,
            'X_test_scaled': X_test_scaled,
            'y_test': y_test,
            'sens_test': sens_test,
            'explainer': explainer
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

models = load_models()

if models is None:
    st.stop()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=100)
    st.title("⚙️ Controls")
    
    # Model selection
    model_choice = st.radio(
        "Select Model",
        ["Fair Model (Demographic Parity)", "Baseline Model (Unconstrained)"],
        help="Choose which model to use for predictions"
    )
    
    # Input features
    st.subheader("📝 Input Features")
    age = st.slider("Age", 18, 80, 35)
    education = st.slider("Education Years", 1, 16, 13)
    hours = st.slider("Hours/Week", 10, 80, 40)
    gender = st.selectbox("Gender", ["Female", "Male"])
    race = st.selectbox("Race", ["Non-White", "White"])
    
    # Convert to numeric
    gender_num = 1 if gender == "Male" else 0
    race_num = 1 if race == "White" else 0
    
    # Analysis options
    st.subheader("🔬 Analysis Options")
    show_shap = st.checkbox("Show SHAP Analysis", True)
    track_energy = st.checkbox("Track Energy Consumption", True)
    
    # Predict button
    predict_btn = st.button("🚀 Run Prediction", type="primary", use_container_width=True)

# Main content area
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model Type", "Fair Model" if "Fair" in model_choice else "Baseline", 
              delta="⚖️ Fairness Constrained" if "Fair" in model_choice else "📈 Max Accuracy")

with col2:
    st.metric("Features Used", "5", delta="Age, Edu, Hours, Gender, Race")

with col3:
    st.metric("Protected Attribute", "Gender (Male/Female)")

with col4:
    st.metric("Fairness Metric", "Demographic Parity")

# Prediction and Analysis
if predict_btn:
    # Prepare features
    features = np.array([[age, education, hours, gender_num, race_num]])
    features_scaled = models['scaler'].transform(features)
    
    # Energy tracking
    if track_energy:
        tracker = EmissionsTracker(save_to_file=False)
        tracker.start()
        start_time = time.time()
        start_cpu = psutil.cpu_percent(interval=None)
        start_memory = psutil.virtual_memory().percent
    
    # Make prediction
    if "Fair" in model_choice:
        model = models['fair_model']
        if hasattr(model, '_model'):
            prediction = model._model.predict(features_scaled)[0]
            probability = model._model.predict_proba(features_scaled)[0][1]
        else:
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0][1]
    else:
        model = models['baseline_model']
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
    
    # Energy tracking results
    if track_energy:
        end_time = time.time()
        end_cpu = psutil.cpu_percent(interval=None)
        end_memory = psutil.virtual_memory().percent
        emissions = tracker.stop()
        
        inference_time = end_time - start_time
        cpu_usage = (start_cpu + end_cpu) / 2
        memory_usage = (start_memory + end_memory) / 2
    
    # Results Dashboard
    st.markdown("---")
    st.header("📊 Prediction Results")
    
    # Metrics row
    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
    
    with res_col1:
        if prediction == 1:
            st.markdown("<div class='metric-card'><h2>💰 HIGH INCOME</h2><p>>50K/year</p></div>", 
                       unsafe_allow_html=True)
        else:
            st.markdown("<div class='metric-card' style='background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);'>"
                       "<h2>📉 LOW INCOME</h2><p><=50K/year</p></div>", unsafe_allow_html=True)
    
    with res_col2:
        st.metric("Confidence", f"{probability:.1%}")
        st.progress(probability)
    
    with res_col3:
        if track_energy:
            st.metric("Inference Time", f"{inference_time*1000:.2f} ms")
            st.metric("CO₂ Emissions", f"{emissions:.6f} kg")
    
    with res_col4:
        # Fairness check for this prediction
        if gender == "Female":
            if prediction == 1:
                fairness_status = "⚖️ Fair (Positive outcome for protected group)"
                fairness_color = "fairness-good"
            else:
                fairness_status = "⚠️ Potential bias detected"
                fairness_color = "fairness-bad"
        else:
            fairness_status = "✅ Privileged group prediction"
            fairness_color = ""
        
        st.markdown(f"**Fairness Check:**")
        st.markdown(f"<p class='{fairness_color}'>{fairness_status}</p>", unsafe_allow_html=True)
    
    # SHAP Analysis
    if show_shap:
        st.markdown("---")
        st.header("🧠 SHAP Explainability Analysis")
        
        with st.spinner("Generating SHAP explanations..."):
            # Calculate SHAP values
            explainer = models['explainer']
            
            if "Fair" in model_choice:
                shap_values = explainer.shap_values(features_scaled)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_values = explainer.shap_values(features_scaled)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            # Create SHAP dataframe
            shap_df = pd.DataFrame({
                'Feature': models['feature_names'],
                'SHAP Value': shap_values[0],
                'Feature Value': [age, education, hours, gender_num, race_num]
            })
            shap_df['Absolute SHAP'] = np.abs(shap_df['SHAP Value'])
            shap_df = shap_df.sort_values('Absolute SHAP', ascending=True)
            
            # Create two columns for SHAP visualization
            shap_col1, shap_col2 = st.columns([2, 1])
            
            with shap_col1:
                # Horizontal bar chart for SHAP values
                fig = go.Figure()
                
                colors = ['#00cc96' if x > 0 else '#ef553b' for x in shap_df['SHAP Value']]
                
                fig.add_trace(go.Bar(
                    y=shap_df['Feature'],
                    x=shap_df['SHAP Value'],
                    orientation='h',
                    marker_color=colors,
                    text=shap_df['SHAP Value'].round(3),
                    textposition='outside',
                    name='SHAP Value'
                ))
                
                fig.update_layout(
                    title="Feature Contributions to Prediction",
                    xaxis_title="SHAP Value (impact on model output)",
                    yaxis_title="Feature",
                    height=400,
                    showlegend=False,
                    hovermode='y'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with shap_col2:
                st.markdown("<div class='explanation-box'>", unsafe_allow_html=True)
                st.markdown("**📌 Interpretation:**")
                st.markdown("""
                - **Positive SHAP** (green) → Increases chance of HIGH income
                - **Negative SHAP** (red) → Decreases chance of HIGH income
                - **Larger bars** → More important for this prediction
                """)
                
                # Find top contributing feature
                top_feature = shap_df.iloc[-1]
                st.markdown(f"**🔍 Top contributor:**")
                st.markdown(f"**{top_feature['Feature']}** (value: {top_feature['Feature Value']})")
                st.markdown(f"contributed **{abs(top_feature['SHAP Value']):.3f}** to the prediction")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Detailed feature table
            with st.expander("📋 View Detailed SHAP Values"):
                st.dataframe(
                    shap_df[['Feature', 'Feature Value', 'SHAP Value', 'Absolute SHAP']]
                    .style.format({
                        'SHAP Value': '{:.4f}',
                        'Absolute SHAP': '{:.4f}'
                    })
                    .background_gradient(subset=['SHAP Value'], cmap='RdBu')
                )
    
    # Energy and Resource Analysis
    if track_energy:
        st.markdown("---")
        st.header("⚡ Energy & Resource Tracking")
        
        en_col1, en_col2, en_col3 = st.columns(3)
        
        with en_col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=inference_time*1000,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Inference Time (ms)"},
                gauge={'axis': {'range': [None, 50]},
                       'bar': {'color': "#1E88E5"},
                       'steps': [
                           {'range': [0, 20], 'color': "lightgray"},
                           {'range': [20, 50], 'color': "gray"}],
                       'threshold': {
                           'line': {'color': "red", 'width': 4},
                           'thickness': 0.75,
                           'value': 30}}))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        with en_col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=cpu_usage,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "CPU Usage (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "#00cc96"},
                       'steps': [
                           {'range': [0, 50], 'color': "lightgray"},
                           {'range': [50, 100], 'color': "gray"}]}))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        with en_col3:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=memory_usage,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Memory Usage (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "#FF6B6B"},
                       'steps': [
                           {'range': [0, 50], 'color': "lightgray"},
                           {'range': [50, 100], 'color': "gray"}]}))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        # CO2 emissions
        st.subheader(f"🌍 Carbon Footprint: **{emissions:.6f} kg CO₂** per inference")
        st.caption(f"Equivalent to approximately {emissions*1000:.2f} grams of CO₂")

# Model Comparison Section
st.markdown("---")
st.header("📈 Model Performance Comparison")

# Load test predictions for both models
if models:
    # Get predictions on test set
    fair_pred = models['fair_model'].predict(models['X_test_scaled'])
    baseline_pred = models['baseline_model'].predict(models['X_test_scaled'])
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score
    from fairlearn.metrics import demographic_parity_difference
    
    fair_acc = accuracy_score(models['y_test'], fair_pred)
    baseline_acc = accuracy_score(models['y_test'], baseline_pred)
    
    fair_gap = demographic_parity_difference(
        models['y_test'], fair_pred, 
        sensitive_features=models['sens_test']
    )
    baseline_gap = demographic_parity_difference(
        models['y_test'], baseline_pred,
        sensitive_features=models['sens_test']
    )
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Model': ['Fair Model', 'Baseline'],
        'Accuracy': [fair_acc, baseline_acc],
        'Fairness Gap': [fair_gap, baseline_gap]
    })
    
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        # Accuracy comparison
        fig = px.bar(comparison_df, x='Model', y='Accuracy', 
                     title='Accuracy Comparison',
                     color='Model',
                     color_discrete_map={'Fair Model': '#1E88E5', 'Baseline': '#FF6B6B'})
        fig.update_layout(showlegend=False, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    
    with comp_col2:
        # Fairness gap comparison (lower is better)
        fig = px.bar(comparison_df, x='Model', y='Fairness Gap',
                     title='Fairness Gap Comparison (Lower is Better)',
                     color='Model',
                     color_discrete_map={'Fair Model': '#1E88E5', 'Baseline': '#FF6B6B'})
        fig.update_layout(showlegend=False, yaxis_range=[0, max(0.2, baseline_gap*1.1)])
        st.plotly_chart(fig, use_container_width=True)
    
    # Fairness improvement
    improvement = baseline_gap - fair_gap
    st.metric(
        "Fairness Improvement",
        f"{improvement:.3f}",
        delta=f"{(improvement/baseline_gap)*100:.1f}% reduction in disparity",
        delta_color="normal"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    Built with ❤️ using Streamlit | Fair AI Dashboard v2.0<br>
    Features: SHAP Explainability, Energy Tracking, Fairness Metrics
</div>
""", unsafe_allow_html=True)
