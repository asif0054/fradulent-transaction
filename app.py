"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║        STEP 6: Streamlit Dashboard — Fraud Detection Web Application           ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  This is the MAIN ENTRY POINT for the web application.                         ║
║                                                                                ║
║  FEATURES:                                                                     ║
║  ─────────                                                                     ║
║  📊 DASHBOARD TAB — Overview metrics, class distribution, transaction charts   ║
║  🔮 PREDICTION TAB — Enter transaction details, get real-time fraud prediction ║
║  📈 ANALYSIS TAB — ROC curves, confusion matrices, feature importance          ║
║  🤖 MODEL COMPARISON — Side-by-side evaluation of all three models             ║
║                                                                                ║
║  RUN THE APP:                                                                  ║
║    streamlit run app.py                                                        ║
║                                                                                ║
║  The app will open at http://localhost:8501                                     ║
║                                                                                ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, accuracy_score
)
import joblib
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# ADD PROJECT ROOT TO PATH
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from models.preprocessing import load_and_preprocess, load_data, clean_data, scale_features
from models.anomaly_detection import run_isolation_forest, run_lof
from models.xgboost_model import train_xgboost, predict_xgboost

# ──────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="💳 Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Premium Dark Theme
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global Styles ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* ── Metric Cards ──────────────────────────────────────── */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(108, 99, 255, 0.2);
    }
    .metric-value {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6C63FF, #FF6584);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 8px 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 500;
    }
    
    /* ── Header ────────────────────────────────────────────── */
    .main-header {
        text-align: center;
        padding: 20px 0 10px 0;
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6C63FF 0%, #FF6584 50%, #6C63FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .main-header p {
        color: #8892b0;
        font-size: 1.1rem;
        margin-top: 8px;
    }
    
    /* ── Prediction Result Cards ───────────────────────────── */
    .result-fraud {
        background: linear-gradient(135deg, #2d1b1b 0%, #3d1515 100%);
        border: 2px solid #ff4444;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        animation: pulse-danger 2s infinite;
    }
    .result-legit {
        background: linear-gradient(135deg, #1b2d1b 0%, #153d15 100%);
        border: 2px solid #00C851;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
    }
    @keyframes pulse-danger {
        0%, 100% { box-shadow: 0 0 10px rgba(255, 68, 68, 0.3); }
        50% { box-shadow: 0 0 25px rgba(255, 68, 68, 0.6); }
    }
    
    /* ── Sidebar Styles ────────────────────────────────────── */
    .sidebar-info {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        border-left: 3px solid #6C63FF;
    }
    
    /* ── Tab Styling ───────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 500;
    }
    
    /* ── Divider ───────────────────────────────────────────── */
    .gradient-divider {
        height: 3px;
        background: linear-gradient(90deg, #6C63FF, #FF6584, #6C63FF);
        border: none;
        border-radius: 2px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

@st.cache_data
def load_dataset():
    """Load and preprocess the credit card dataset (cached)."""
    data_path = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")
    
    if not os.path.exists(data_path):
        st.error("⚠️ Dataset not found! Run `python data/download_data.py` first.")
        st.stop()
    
    df = pd.read_csv(data_path)
    return df


@st.cache_resource
def train_all_models():
    """Train all models and cache the results."""
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess(
        os.path.join(PROJECT_ROOT, "data", "creditcard.csv"),
        use_smote=True
    )
    
    # Also get unbalanced data for anomaly detection
    X_train_raw, X_test_raw, y_train_raw, y_test_raw, _, _ = load_and_preprocess(
        os.path.join(PROJECT_ROOT, "data", "creditcard.csv"),
        use_smote=False
    )
    
    # Train XGBoost
    xgb_results = train_xgboost(X_train, y_train, X_test, y_test, save=True)
    
    # Train Isolation Forest (on unbalanced data)
    iso_results = run_isolation_forest(X_train_raw, X_test_raw, y_test_raw, save=True)
    
    # Train LOF (on unbalanced data)
    lof_results = run_lof(X_train_raw, X_test_raw, y_test_raw)
    
    return {
        "xgboost": xgb_results,
        "isolation_forest": iso_results,
        "lof": lof_results,
        "X_test": X_test,
        "y_test": y_test,
        "X_test_raw": X_test_raw,
        "y_test_raw": y_test_raw,
        "scaler": scaler,
        "feature_names": feature_names
    }


def create_metric_card(label, value, icon="📊"):
    """Create a styled metric card."""
    return f"""
    <div class="metric-card">
        <div style="font-size: 1.8rem; margin-bottom: 4px;">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


# ══════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════

def main():
    """Main application entry point."""
    
    # ── Header ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Fraud Detection Dashboard</h1>
        <p>Real-time credit card transaction fraud detection powered by ML</p>
    </div>
    <div class="gradient-divider"></div>
    """, unsafe_allow_html=True)
    
    # ── Sidebar ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Control Panel")
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        
        # Check for dataset
        data_path = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")
        
        if not os.path.exists(data_path):
            st.warning("📂 Dataset not found!")
            if st.button("🔄 Generate Dataset", use_container_width=True):
                with st.spinner("Generating synthetic dataset..."):
                    sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))
                    from data.download_data import generate_dataset
                    generate_dataset()
                st.success("✅ Dataset generated!")
                st.rerun()
            st.stop()
        
        st.markdown("""
        <div class="sidebar-info">
            <strong>🟢 Dataset Loaded</strong><br>
            <small>Credit Card Transactions</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Model training button
        st.markdown("### 🤖 Model Training")
        train_button = st.button("🚀 Train All Models", use_container_width=True, type="primary")
        
        if train_button or 'models_trained' in st.session_state:
            if train_button:
                with st.spinner("⏳ Training models... This may take a minute."):
                    results = train_all_models()
                    st.session_state['results'] = results
                    st.session_state['models_trained'] = True
                st.success("✅ All models trained!")
            
        st.markdown("---")
        st.markdown("""
        ### 📖 About
        This dashboard uses three models:
        - 🌲 **Isolation Forest**
        - 🔍 **Local Outlier Factor**  
        - 🚀 **XGBoost Classifier**
        
        Built with Streamlit, Scikit-Learn & XGBoost
        """)
    
    # ── Load data for display ──────────────────────────────────────────
    df = load_dataset()
    
    # ── Main Tabs ──────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔮 Predict Transaction", "📈 Model Analysis"])
    
    # ══════════════════════════════════════════════════════════════════
    # TAB 1: DASHBOARD — Overview & Data Exploration
    # ══════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("### 📊 Dataset Overview")
        
        # ── Metric Cards Row ───────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        
        total_txn = len(df)
        fraud_txn = (df['Class'] == 1).sum()
        legit_txn = (df['Class'] == 0).sum()
        fraud_rate = fraud_txn / total_txn * 100
        
        with col1:
            st.markdown(create_metric_card("Total Transactions", f"{total_txn:,}", "💳"), 
                        unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric_card("Legitimate", f"{legit_txn:,}", "✅"), 
                        unsafe_allow_html=True)
        with col3:
            st.markdown(create_metric_card("Fraudulent", f"{fraud_txn:,}", "🔴"), 
                        unsafe_allow_html=True)
        with col4:
            st.markdown(create_metric_card("Fraud Rate", f"{fraud_rate:.2f}%", "📉"), 
                        unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ── Charts Row ────────────────────────────────────────────────
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("#### 🍩 Class Distribution")
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Legitimate', 'Fraudulent'],
                values=[legit_txn, fraud_txn],
                hole=0.55,
                marker=dict(colors=['#6C63FF', '#FF6584'],
                           line=dict(color='#0E1117', width=3)),
                textfont=dict(size=14, color='white'),
                textinfo='label+percent',
            )])
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=False,
                height=350,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_right:
            st.markdown("#### 💰 Transaction Amount Distribution")
            fig_amount = go.Figure()
            fig_amount.add_trace(go.Histogram(
                x=df[df['Class']==0]['Amount'], name='Legitimate',
                marker_color='#6C63FF', opacity=0.7, nbinsx=50
            ))
            fig_amount.add_trace(go.Histogram(
                x=df[df['Class']==1]['Amount'], name='Fraudulent',
                marker_color='#FF6584', opacity=0.7, nbinsx=50
            ))
            fig_amount.update_layout(
                barmode='overlay',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis_title='Amount ($)',
                yaxis_title='Count',
                height=350,
                margin=dict(t=20, b=40, l=40, r=20),
                legend=dict(x=0.7, y=0.95, bgcolor='rgba(0,0,0,0.5)')
            )
            fig_amount.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
            fig_amount.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
            st.plotly_chart(fig_amount, use_container_width=True)
        
        # ── Transaction Timeline ───────────────────────────────────────
        st.markdown("#### ⏰ Transactions Over Time")
        df_time = df.copy()
        df_time['Time_Hours'] = df_time['Time'] / 3600
        
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=df_time[df_time['Class']==0]['Time_Hours'],
            y=df_time[df_time['Class']==0]['Amount'],
            mode='markers', name='Legitimate',
            marker=dict(color='#6C63FF', size=3, opacity=0.3)
        ))
        fig_time.add_trace(go.Scatter(
            x=df_time[df_time['Class']==1]['Time_Hours'],
            y=df_time[df_time['Class']==1]['Amount'],
            mode='markers', name='Fraudulent',
            marker=dict(color='#FF6584', size=8, opacity=0.8,
                       line=dict(width=1, color='white'))
        ))
        fig_time.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis_title='Time (Hours)',
            yaxis_title='Amount ($)',
            height=400,
            margin=dict(t=20, b=40, l=40, r=20),
            legend=dict(x=0.85, y=0.95, bgcolor='rgba(0,0,0,0.5)')
        )
        fig_time.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
        fig_time.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
        st.plotly_chart(fig_time, use_container_width=True)
        
        # ── Data Sample ───────────────────────────────────────────────
        with st.expander("📋 View Raw Data Sample"):
            st.dataframe(df.head(20), use_container_width=True)
            st.markdown(f"**Columns:** {', '.join(df.columns)}")
    
    # ══════════════════════════════════════════════════════════════════
    # TAB 2: PREDICTION — Real-Time Fraud Detection
    # ══════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### 🔮 Real-Time Transaction Prediction")
        st.markdown("Enter transaction details below to check if it's fraudulent.")
        
        if 'models_trained' not in st.session_state:
            st.warning("⚠️ Please train the models first using the sidebar button!")
        else:
            results = st.session_state['results']
            
            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
            
            # ── Input Method Selection ─────────────────────────────────
            input_method = st.radio(
                "Choose input method:",
                ["📝 Manual Input", "🎲 Random Sample from Dataset", "📋 Use Test Sample"],
                horizontal=True
            )
            
            feature_names = results['feature_names']
            
            if input_method == "📝 Manual Input":
                st.markdown("#### Enter Feature Values")
                st.info("💡 V1–V28 are PCA-transformed features. Use values between -5 and 5 for realistic inputs.")
                
                input_values = {}
                cols = st.columns(4)
                for i, feat in enumerate(feature_names):
                    with cols[i % 4]:
                        default_val = 0.0
                        input_values[feat] = st.number_input(
                            feat, value=default_val, step=0.1,
                            format="%.4f", key=f"input_{feat}"
                        )
                
                input_df = pd.DataFrame([input_values])
            
            elif input_method == "🎲 Random Sample from Dataset":
                if st.button("🎲 Generate Random Transaction", type="primary"):
                    X_test = results['X_test']
                    idx = np.random.randint(0, len(X_test))
                    if hasattr(X_test, 'iloc'):
                        sample = X_test.iloc[idx:idx+1]
                    else:
                        sample = pd.DataFrame([X_test[idx]], columns=feature_names)
                    st.session_state['random_sample'] = sample
                    true_label = results['y_test'].iloc[idx] if hasattr(results['y_test'], 'iloc') else results['y_test'][idx]
                    st.session_state['random_true_label'] = true_label
                
                if 'random_sample' in st.session_state:
                    input_df = st.session_state['random_sample']
                    true_label = st.session_state['random_true_label']
                    st.info(f"📌 True label: {'🔴 Fraud' if true_label == 1 else '🟢 Legitimate'}")
                    with st.expander("View feature values"):
                        st.dataframe(input_df, use_container_width=True)
                else:
                    input_df = None
            
            else:  # Test Sample
                X_test = results['X_test']
                sample_idx = st.slider("Select test sample index", 0, len(X_test)-1, 0)
                if hasattr(X_test, 'iloc'):
                    input_df = X_test.iloc[sample_idx:sample_idx+1]
                else:
                    input_df = pd.DataFrame([X_test[sample_idx]], columns=feature_names)
                true_label = results['y_test'].iloc[sample_idx] if hasattr(results['y_test'], 'iloc') else results['y_test'][sample_idx]
                st.info(f"📌 True label: {'🔴 Fraud' if true_label == 1 else '🟢 Legitimate'}")
            
            # ── Run Prediction ─────────────────────────────────────────
            if input_df is not None:
                st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
                
                if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
                    with st.spinner("Analyzing transaction..."):
                        xgb_model = results['xgboost']['model']
                        prediction = predict_xgboost(xgb_model, input_df.values[0])
                    
                    # Display Result
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    col_res1, col_res2 = st.columns([1, 1])
                    
                    with col_res1:
                        if prediction['prediction'] == 1:
                            st.markdown(f"""
                            <div class="result-fraud">
                                <h2 style="color: #ff4444; margin: 0;">🚨 FRAUD DETECTED</h2>
                                <p style="font-size: 3rem; margin: 10px 0; font-weight: 700; color: #ff4444;">
                                    {prediction['fraud_probability']*100:.1f}%
                                </p>
                                <p style="color: #ff9999;">Confidence of Fraud</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-legit">
                                <h2 style="color: #00C851; margin: 0;">✅ LEGITIMATE</h2>
                                <p style="font-size: 3rem; margin: 10px 0; font-weight: 700; color: #00C851;">
                                    {prediction['legit_probability']*100:.1f}%
                                </p>
                                <p style="color: #99ffbb;">Confidence of Legitimacy</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col_res2:
                        # Confidence gauge
                        fraud_pct = prediction['fraud_probability'] * 100
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=fraud_pct,
                            title={'text': "Fraud Risk Score", 'font': {'size': 18, 'color': 'white'}},
                            number={'suffix': '%', 'font': {'color': 'white'}},
                            gauge={
                                'axis': {'range': [0, 100], 'tickcolor': 'white'},
                                'bar': {'color': '#FF6584'},
                                'bgcolor': 'rgba(0,0,0,0)',
                                'steps': [
                                    {'range': [0, 30], 'color': 'rgba(46, 204, 113, 0.3)'},
                                    {'range': [30, 70], 'color': 'rgba(241, 196, 15, 0.3)'},
                                    {'range': [70, 100], 'color': 'rgba(231, 76, 60, 0.3)'}
                                ],
                                'threshold': {
                                    'line': {'color': 'white', 'width': 3},
                                    'thickness': 0.8,
                                    'value': 50
                                }
                            }
                        ))
                        fig_gauge.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            font={'color': 'white'},
                            height=300,
                            margin=dict(t=60, b=20, l=30, r=30)
                        )
                        st.plotly_chart(fig_gauge, use_container_width=True)
    
    # ══════════════════════════════════════════════════════════════════
    # TAB 3: MODEL ANALYSIS — ROC, Confusion Matrix, Comparison
    # ══════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### 📈 Model Analysis & Comparison")
        
        if 'models_trained' not in st.session_state:
            st.warning("⚠️ Please train the models first using the sidebar button!")
        else:
            results = st.session_state['results']
            
            # ── XGBoost Metrics Row ────────────────────────────────────
            st.markdown("#### 🚀 XGBoost Performance")
            
            xgb = results['xgboost']
            y_test = results['y_test']
            y_test_raw = results['y_test_raw']
            
            prec = precision_score(y_test, xgb['predictions'], zero_division=0)
            rec = recall_score(y_test, xgb['predictions'], zero_division=0)
            f1 = f1_score(y_test, xgb['predictions'], zero_division=0)
            acc = accuracy_score(y_test, xgb['predictions'])
            
            fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb['probabilities'])
            auc_xgb = auc(fpr_xgb, tpr_xgb)
            
            mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
            with mcol1:
                st.metric("Accuracy", f"{acc:.4f}")
            with mcol2:
                st.metric("Precision", f"{prec:.4f}")
            with mcol3:
                st.metric("Recall", f"{rec:.4f}")
            with mcol4:
                st.metric("F1 Score", f"{f1:.4f}")
            with mcol5:
                st.metric("AUC-ROC", f"{auc_xgb:.4f}")
            
            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
            
            # ── ROC Curves ─────────────────────────────────────────────
            col_roc, col_cm = st.columns(2)
            
            with col_roc:
                st.markdown("#### 📈 ROC Curves (All Models)")
                
                fig_roc = go.Figure()
                
                # XGBoost ROC
                fig_roc.add_trace(go.Scatter(
                    x=fpr_xgb, y=tpr_xgb,
                    name=f'XGBoost (AUC={auc_xgb:.4f})',
                    line=dict(color='#6C63FF', width=3),
                    fill='tozeroy', fillcolor='rgba(108, 99, 255, 0.1)'
                ))
                
                # Isolation Forest ROC (using scores)
                iso_scores = results['isolation_forest']['scores']
                fpr_iso, tpr_iso, _ = roc_curve(y_test_raw, -iso_scores)  # Negate: lower = more anomalous
                auc_iso = auc(fpr_iso, tpr_iso)
                fig_roc.add_trace(go.Scatter(
                    x=fpr_iso, y=tpr_iso,
                    name=f'Isolation Forest (AUC={auc_iso:.4f})',
                    line=dict(color='#FF6584', width=3)
                ))
                
                # LOF ROC
                lof_scores = results['lof']['scores']
                fpr_lof, tpr_lof, _ = roc_curve(y_test_raw, -lof_scores)
                auc_lof = auc(fpr_lof, tpr_lof)
                fig_roc.add_trace(go.Scatter(
                    x=fpr_lof, y=tpr_lof,
                    name=f'LOF (AUC={auc_lof:.4f})',
                    line=dict(color='#2ECC71', width=3)
                ))
                
                # Random baseline
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    name='Random',
                    line=dict(color='gray', width=1, dash='dash')
                ))
                
                fig_roc.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    height=450,
                    margin=dict(t=20, b=40, l=40, r=20),
                    legend=dict(x=0.4, y=0.05, bgcolor='rgba(0,0,0,0.7)',
                               font=dict(size=11))
                )
                fig_roc.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
                fig_roc.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
                st.plotly_chart(fig_roc, use_container_width=True)
            
            with col_cm:
                st.markdown("#### 🔲 XGBoost Confusion Matrix")
                
                cm = confusion_matrix(y_test, xgb['predictions'])
                
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm[::-1],
                    x=['Legitimate', 'Fraud'],
                    y=['Fraud', 'Legitimate'],
                    colorscale=[[0, '#1a1a2e'], [0.5, '#6C63FF'], [1, '#FF6584']],
                    text=cm[::-1],
                    texttemplate='<b>%{text}</b>',
                    textfont=dict(size=22, color='white'),
                    showscale=False,
                    hovertemplate="Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>"
                ))
                fig_cm.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis_title='Predicted Label',
                    yaxis_title='True Label',
                    height=450,
                    margin=dict(t=20, b=40, l=60, r=20)
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            
            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
            
            # ── Feature Importance ─────────────────────────────────────
            st.markdown("#### 🏆 Top Feature Importance (XGBoost)")
            
            importance = xgb['feature_importance']
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
            feat_names = [f[0] for f in sorted_imp]
            feat_scores = [f[1] for f in sorted_imp]
            
            fig_imp = go.Figure(go.Bar(
                x=feat_scores,
                y=feat_names,
                orientation='h',
                marker=dict(
                    color=feat_scores,
                    colorscale=[[0, '#6C63FF'], [0.5, '#a855f7'], [1, '#FF6584']],
                    line=dict(width=0)
                ),
                text=[f'{s:.4f}' for s in feat_scores],
                textposition='outside',
                textfont=dict(color='white', size=11)
            ))
            fig_imp.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis_title='Importance Score',
                height=500,
                margin=dict(t=20, b=40, l=100, r=60),
                yaxis=dict(autorange='reversed')
            )
            fig_imp.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
            st.plotly_chart(fig_imp, use_container_width=True)
            
            # ── Model Comparison Table ─────────────────────────────────
            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
            st.markdown("#### 📋 Model Comparison Summary")
            
            iso_res = results['isolation_forest']
            lof_res = results['lof']
            
            comparison_data = {
                "Model": ["🚀 XGBoost", "🌲 Isolation Forest", "🔍 LOF"],
                "Accuracy": [
                    f"{acc:.4f}",
                    f"{iso_res['accuracy']:.4f}",
                    f"{lof_res['accuracy']:.4f}"
                ],
                "AUC-ROC": [
                    f"{auc_xgb:.4f}",
                    f"{auc_iso:.4f}",
                    f"{auc_lof:.4f}"
                ],
                "Type": [
                    "Supervised",
                    "Unsupervised",
                    "Unsupervised"
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # ── Classification Reports ─────────────────────────────────
            with st.expander("📄 Detailed Classification Reports"):
                rcol1, rcol2, rcol3 = st.columns(3)
                with rcol1:
                    st.markdown("**XGBoost**")
                    st.text(xgb['report'])
                with rcol2:
                    st.markdown("**Isolation Forest**")
                    st.text(iso_res['report'])
                with rcol3:
                    st.markdown("**LOF**")
                    st.text(lof_res['report'])


# ──────────────────────────────────────────────────────────────────────
# RUN THE APP
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
