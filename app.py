import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import shap
import lime
import lime.lime_tabular
import joblib
import io
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Overview", "EDA", "Model Training", "Prediction", "Explainability"]
)

# Theme toggle
st.sidebar.markdown("---")
theme = st.sidebar.radio("Theme:", ["Light", "Dark"])
st.session_state.theme = theme.lower()

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================
if page == "Overview":
    st.title("Credit Card Fraud Detection System")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Project Overview
        
        Credit card fraud is a critical issue affecting millions of transactions daily. 
        This system uses machine learning to detect fraudulent transactions with high accuracy.
        
        **Key Objectives:**
        - Classify transactions as fraudulent or legitimate
        - Identify patterns in fraudulent behavior
        - Provide explainable predictions
        - Enable real-time fraud detection
        
        **Dataset Information:**
        - Source: Kaggle Credit Card Fraud Detection Dataset
        - Transactions: 284,807 records
        - Features: 30 (28 PCA-transformed + Amount + Time)
        - Fraud Rate: ~0.17% (highly imbalanced)
        - Time Period: 2 days of September 2013
        """)
    
    with col2:
        st.info("""
        **Quick Stats**
        - Total Transactions: 284,807
        - Fraudulent: 492 (0.17%)
        - Legitimate: 284,315 (99.83%)
        - Features: 30
        """)
    
    st.markdown("---")
    
    # Dataset upload section
    st.subheader("Load Dataset")
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Upload CSV dataset", type=['csv'])
        if uploaded_file is not None:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.success("Dataset loaded successfully!")
    
    with col2:
        if st.button("Load Sample Dataset"):
            # Create sample dataset
            np.random.seed(42)
            n_samples = 1000
            n_features = 28
            
            X = np.random.randn(n_samples, n_features)
            y = np.random.binomial(1, 0.02, n_samples)
            
            data = pd.DataFrame(X, columns=[f'V{i+1}' for i in range(n_features)])
            data['Amount'] = np.random.exponential(100, n_samples)
            data['Time'] = np.random.randint(0, 172800, n_samples)
            data['Class'] = y
            
            st.session_state.data = data
            st.success("Sample dataset loaded!")
    
    # Display dataset info
    if st.session_state.data is not None:
        st.subheader("Dataset Preview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(st.session_state.data))
        with col2:
            st.metric("Total Features", len(st.session_state.data.columns))
        with col3:
            st.metric("Missing Values", st.session_state.data.isnull().sum().sum())
        with col4:
            fraud_count = (st.session_state.data['Class'] == 1).sum() if 'Class' in st.session_state.data.columns else 0
            st.metric("Fraud Cases", fraud_count)
        
        st.dataframe(st.session_state.data.head(10), use_container_width=True)
        
        # Class distribution
        if 'Class' in st.session_state.data.columns:
            st.subheader("Class Distribution")
            class_dist = st.session_state.data['Class'].value_counts()
            
            fig = go.Figure(data=[
                go.Pie(labels=['Legitimate', 'Fraudulent'], 
                       values=[class_dist[0], class_dist[1]],
                       marker=dict(colors=['#2ecc71', '#e74c3c']))
            ])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 2: EDA
# ============================================================================
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    
    if st.session_state.data is None:
        st.warning("Please load a dataset first from the Overview page.")
    else:
        data = st.session_state.data
        
        # Data statistics
        st.subheader("Data Statistics")
        st.dataframe(data.describe(), use_container_width=True)
        
        # Missing values
        st.subheader("Missing Values")
        missing = data.isnull().sum()
        if missing.sum() == 0:
            st.success("No missing values found!")
        else:
            st.dataframe(missing[missing > 0])
        
        # Distribution plots
        st.subheader("Feature Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Amount' in data.columns:
                fig = px.histogram(data, x='Amount', nbins=50, 
                                  title='Amount Distribution',
                                  color_discrete_sequence=['#3498db'])
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Time' in data.columns:
                fig = px.histogram(data, x='Time', nbins=50,
                                  title='Time Distribution',
                                  color_discrete_sequence=['#9b59b6'])
                st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr_matrix = data[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu'
        ))
        fig.update_layout(height=600, width=800)
        st.plotly_chart(fig, use_container_width=True)
        
        # Fraud vs Normal comparison
        if 'Class' in data.columns:
            st.subheader("Fraud vs Normal Transactions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Amount' in data.columns:
                    fig = px.box(data, x='Class', y='Amount',
                                title='Amount by Transaction Type',
                                labels={'Class': 'Transaction Type'},
                                color_discrete_sequence=['#2ecc71', '#e74c3c'])
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'Time' in data.columns:
                    fig = px.box(data, x='Class', y='Time',
                                title='Time by Transaction Type',
                                labels={'Class': 'Transaction Type'},
                                color_discrete_sequence=['#2ecc71', '#e74c3c'])
                    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 3: MODEL TRAINING
# ============================================================================
elif page == "Model Training":
    st.title("Model Training & Evaluation")
    
    if st.session_state.data is None:
        st.warning("Please load a dataset first from the Overview page.")
    else:
        data = st.session_state.data
        
        # Data preprocessing
        st.subheader("Data Preprocessing")
        
        if st.button("Preprocess Data"):
            with st.spinner("Preprocessing data..."):
                # Separate features and target
                X = data.drop('Class', axis=1)
                y = data['Class']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Store in session state
                st.session_state.X_train = X_train_scaled
                st.session_state.X_test = X_test_scaled
                st.session_state.y_train = y_train.values
                st.session_state.y_test = y_test.values
                st.session_state.scaler = scaler
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Training Samples", len(X_train))
                with col2:
                    st.metric("Test Samples", len(X_test))
                with col3:
                    st.metric("Features", X_train.shape[1])
                with col4:
                    st.metric("Train Fraud %", f"{(y_train.sum()/len(y_train)*100):.2f}%")
                
                st.success("Data preprocessing completed!")
        
        if st.session_state.X_train is not None:
            st.markdown("---")
            
            # Model selection
            st.subheader("Model Selection & Training")
            
            col1, col2 = st.columns(2)
            
            with col1:
                model_choice = st.selectbox(
                    "Select Algorithm:",
                    ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "CatBoost"]
                )
            
            with col2:
                auto_ml = st.checkbox("AutoML Mode (Compare All Models)")
            
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    models = {}
                    results = {}
                    
                    if auto_ml:
                        # Train all models
                        models['Logistic Regression'] = LogisticRegression(max_iter=1000, random_state=42)
                        models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                        models['XGBoost'] = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
                        models['LightGBM'] = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
                        models['CatBoost'] = CatBoostClassifier(iterations=100, verbose=False, random_state=42)
                        
                        for name, model in models.items():
                            model.fit(st.session_state.X_train, st.session_state.y_train)
                            y_pred = model.predict(st.session_state.X_test)
                            y_pred_proba = model.predict_proba(st.session_state.X_test)[:, 1]
                            
                            results[name] = {
                                'accuracy': accuracy_score(st.session_state.y_test, y_pred),
                                'precision': precision_score(st.session_state.y_test, y_pred),
                                'recall': recall_score(st.session_state.y_test, y_pred),
                                'f1': f1_score(st.session_state.y_test, y_pred),
                                'roc_auc': roc_auc_score(st.session_state.y_test, y_pred_proba)
                            }
                        
                        # Display comparison
                        st.subheader("Model Comparison")
                        results_df = pd.DataFrame(results).T
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Select best model
                        best_model_name = results_df['roc_auc'].idxmax()
                        st.session_state.model = models[best_model_name]
                        st.session_state.model_metrics = results[best_model_name]
                        
                        st.success(f"Best Model: {best_model_name} (ROC-AUC: {results[best_model_name]['roc_auc']:.4f})")
                    
                    else:
                        # Train selected model
                        if model_choice == "Logistic Regression":
                            model = LogisticRegression(max_iter=1000, random_state=42)
                        elif model_choice == "Random Forest":
                            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                        elif model_choice == "XGBoost":
                            model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
                        elif model_choice == "LightGBM":
                            model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
                        else:  # CatBoost
                            model = CatBoostClassifier(iterations=100, verbose=False, random_state=42)
                        
                        model.fit(st.session_state.X_train, st.session_state.y_train)
                        
                        # Predictions
                        y_pred = model.predict(st.session_state.X_test)
                        y_pred_proba = model.predict_proba(st.session_state.X_test)[:, 1]
                        
                        # Calculate metrics
                        metrics = {
                            'accuracy': accuracy_score(st.session_state.y_test, y_pred),
                            'precision': precision_score(st.session_state.y_test, y_pred),
                            'recall': recall_score(st.session_state.y_test, y_pred),
                            'f1': f1_score(st.session_state.y_test, y_pred),
                            'roc_auc': roc_auc_score(st.session_state.y_test, y_pred_proba)
                        }
                        
                        st.session_state.model = model
                        st.session_state.model_metrics = metrics
                        
                        st.success(f"{model_choice} trained successfully!")
            
            # Display metrics
            if st.session_state.model is not None:
                st.markdown("---")
                st.subheader("Model Performance Metrics")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                metrics = st.session_state.model_metrics
                
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.4f}")
                with col3:
                    st.metric("Recall", f"{metrics['recall']:.4f}")
                with col4:
                    st.metric("F1-Score", f"{metrics['f1']:.4f}")
                with col5:
                    st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                y_pred = st.session_state.model.predict(st.session_state.X_test)
                cm = confusion_matrix(st.session_state.y_test, y_pred)
                
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Predicted Normal', 'Predicted Fraud'],
                    y=['Actual Normal', 'Actual Fraud'],
                    text=cm,
                    texttemplate='%{text}',
                    colorscale='Blues'
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                # ROC Curve
                st.subheader("ROC Curve")
                y_pred_proba = st.session_state.model.predict_proba(st.session_state.X_test)[:, 1]
                fpr, tpr, _ = roc_curve(st.session_state.y_test, y_pred_proba)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve',
                                        line=dict(color='#3498db', width=2)))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier',
                                        line=dict(color='#95a5a6', width=2, dash='dash')))
                fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate',
                                 yaxis_title='True Positive Rate', height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Model download
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Download Model"):
                        model_bytes = io.BytesIO()
                        joblib.dump(st.session_state.model, model_bytes)
                        model_bytes.seek(0)
                        st.download_button(
                            label="Download Trained Model",
                            data=model_bytes.getvalue(),
                            file_name="fraud_detection_model.pkl",
                            mime="application/octet-stream"
                        )
                
                with col2:
                    if st.button("Download Report"):
                        report = f"""
                        Credit Card Fraud Detection Model Report
                        ==========================================
                        
                        Model Metrics:
                        - Accuracy: {metrics['accuracy']:.4f}
                        - Precision: {metrics['precision']:.4f}
                        - Recall: {metrics['recall']:.4f}
                        - F1-Score: {metrics['f1']:.4f}
                        - ROC-AUC: {metrics['roc_auc']:.4f}
                        """
                        st.download_button(
                            label="Download Report",
                            data=report,
                            file_name="model_report.txt",
                            mime="text/plain"
                        )

# ============================================================================
# PAGE 4: PREDICTION
# ============================================================================
elif page == "Prediction":
    st.title("Fraud Detection Prediction")
    
    if st.session_state.model is None:
        st.warning("Please train a model first from the Model Training page.")
    else:
        st.subheader("Enter Transaction Details")
        
        # Create input form
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
            time = st.number_input("Time (seconds)", min_value=0, value=0)
        
        with col2:
            st.info("Enter V1-V28 features (PCA-transformed)")
        
        # V features input
        v_features = []
        cols = st.columns(4)
        for i in range(28):
            with cols[i % 4]:
                v_val = st.number_input(f"V{i+1}", value=0.0, key=f"v{i+1}")
                v_features.append(v_val)
        
        if st.button("Predict Fraud", use_container_width=True):
            # Prepare input
            input_data = np.array([v_features + [amount, time]])
            input_scaled = st.session_state.scaler.transform(input_data)
            
            # Make prediction
            prediction = st.session_state.model.predict(input_scaled)[0]
            probability = st.session_state.model.predict_proba(input_scaled)[0]
            
            # Display result
            st.markdown("---")
            st.subheader("Prediction Result")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("FRAUD DETECTED")
                    st.metric("Fraud Probability", f"{probability[1]*100:.2f}%")
                else:
                    st.success("LEGITIMATE TRANSACTION")
                    st.metric("Legitimate Probability", f"{probability[0]*100:.2f}%")
            
            with col2:
                fig = go.Figure(data=[
                    go.Bar(x=['Legitimate', 'Fraudulent'],
                          y=[probability[0], probability[1]],
                          marker_color=['#2ecc71', '#e74c3c'])
                ])
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 5: EXPLAINABILITY
# ============================================================================
elif page == "Explainability":
    st.title("Model Explainability")
    
    if st.session_state.model is None:
        st.warning("Please train a model first from the Model Training page.")
    else:
        st.subheader("Feature Importance Analysis")
        
        # Feature importance
        if hasattr(st.session_state.model, 'feature_importances_'):
            importances = st.session_state.model.feature_importances_
            feature_names = [f'V{i+1}' for i in range(28)] + ['Amount', 'Time']
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(15)
            
            fig = px.bar(importance_df, x='Importance', y='Feature',
                        orientation='h', title='Top 15 Important Features',
                        color_discrete_sequence=['#3498db'])
            st.plotly_chart(fig, use_container_width=True)
        
        # SHAP explanation
        st.subheader("SHAP Explanation")
        
        if st.button("Generate SHAP Explanation"):
            with st.spinner("Generating SHAP values..."):
                # Use subset for faster computation
                X_sample = st.session_state.X_test[:100]
                
                explainer = shap.TreeExplainer(st.session_state.model) if hasattr(st.session_state.model, 'feature_importances_') else shap.KernelExplainer(st.session_state.model.predict, X_sample)
                shap_values = explainer.shap_values(X_sample)
                
                # Summary plot
                fig = plt.figure(figsize=(10, 6))
                if isinstance(shap_values, list):
                    shap.summary_plot(shap_values[1], X_sample, show=False)
                else:
                    shap.summary_plot(shap_values, X_sample, show=False)
                st.pyplot(fig)
        
        st.info("SHAP values show how each feature contributes to the model's predictions.")
