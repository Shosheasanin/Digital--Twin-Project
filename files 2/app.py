"""
Water Quality Detection AI App - Streamlit Frontend
Full-featured app with CSV upload, ML prediction, visualization, and LangChain Q&A.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
from io import StringIO

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocess import load_csv, validate_columns, handle_missing_values, EXPECTED_COLUMNS
from train_model import train_model, load_model
from predict import predict_from_dataframe, predict_single_sample, get_prediction_summary
from langchain_agent import create_agent


# ==================== Page Config ====================
st.set_page_config(
    page_title="Water Quality Detection AI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Custom Styling ====================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #0077b6 0%, #00b4d8 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f9ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #0077b6;
    }
    .safe-badge {
        background: #10b981; color: white; padding: 4px 12px;
        border-radius: 12px; font-weight: bold;
    }
    .unsafe-badge {
        background: #ef4444; color: white; padding: 4px 12px;
        border-radius: 12px; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Session State ====================
if "df" not in st.session_state:
    st.session_state.df = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "model" not in st.session_state:
    st.session_state.model = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ==================== Header ====================
st.markdown("""
<div class="main-header">
    <h1>💧 Water Quality Detection AI</h1>
    <p>Upload water data → Get instant safety predictions → Ask questions in plain English</p>
</div>
""", unsafe_allow_html=True)


# ==================== Sidebar ====================
with st.sidebar:
    st.header("⚙️ Configuration")

    # LLM Provider
    llm_provider = st.selectbox(
        "LLM Provider (for chat)",
        ["groq", "openai", "fallback (no API)"],
        help="Groq is free. OpenAI needs a paid key."
    )

    api_key = None
    if llm_provider != "fallback (no API)":
        api_key = st.text_input(
            f"{llm_provider.upper()} API Key",
            type="password",
            help="Stored only in this session"
        )

    st.divider()
    st.header("📊 Model Actions")

    if st.button("🔄 Train New Model", use_container_width=True):
        if st.session_state.df is not None and "Potability" in st.session_state.df.columns:
            with st.spinner("Training model... (~30 seconds)"):
                try:
                    # Save uploaded df temporarily for training
                    temp_path = "data/_temp_training.csv"
                    os.makedirs("data", exist_ok=True)
                    st.session_state.df.to_csv(temp_path, index=False)
                    model, scaler, metrics = train_model(temp_path)
                    st.session_state.model = model
                    st.session_state.scaler = scaler
                    st.success(f"✅ Trained! Accuracy: {metrics['accuracy']:.3f}")
                except Exception as e:
                    st.error(f"Training failed: {e}")
        else:
            st.warning("Upload CSV with 'Potability' column first")

    if st.button("📥 Load Saved Model", use_container_width=True):
        try:
            model, scaler = load_model()
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.success("✅ Model loaded")
        except Exception as e:
            st.error(f"No saved model: {e}")

    st.divider()
    st.caption("💡 Dataset: Kaggle Water Potability")
    st.caption("Expected columns: " + ", ".join(EXPECTED_COLUMNS))


# ==================== Main Tabs ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Upload Data", "📊 Analysis", "🔮 Predictions", "📈 Visualizations", "💬 AI Chat"
])


# ==================== TAB 1: Upload ====================
with tab1:
    st.subheader("Upload Water Quality CSV")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Drop your CSV file here",
            type=["csv"],
            help="Must contain water parameters like pH, Turbidity, etc."
        )

    with col2:
        st.info("**Or** try sample data:")
        if st.button("🧪 Load Sample Data", use_container_width=True):
            # Generate realistic sample
            np.random.seed(42)
            n = 100
            sample = pd.DataFrame({
                "ph": np.random.normal(7.0, 1.5, n).clip(0, 14),
                "Hardness": np.random.normal(200, 40, n),
                "Solids": np.random.normal(22000, 8000, n),
                "Chloramines": np.random.normal(7, 1.5, n),
                "Sulfate": np.random.normal(330, 40, n),
                "Conductivity": np.random.normal(425, 80, n),
                "Organic_carbon": np.random.normal(14, 3, n),
                "Trihalomethanes": np.random.normal(66, 16, n),
                "Turbidity": np.random.normal(4, 0.8, n),
                "Potability": np.random.choice([0, 1], n, p=[0.6, 0.4])
            })
            st.session_state.df = sample
            st.success(f"✅ Loaded {n} sample rows")
            st.rerun()

    if uploaded_file is not None:
        try:
            df = load_csv(uploaded_file)
            st.session_state.df = df
            st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            st.error(f"Could not read file: {e}")

    if st.session_state.df is not None:
        df = st.session_state.df
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        is_valid, missing = validate_columns(df)
        if not is_valid:
            st.warning(f"⚠️ Missing expected columns: {missing}")
        else:
            st.success("✅ All expected columns present")


# ==================== TAB 2: Analysis ====================
with tab2:
    st.subheader("Data Analysis")

    if st.session_state.df is None:
        st.info("👈 Upload data in the first tab")
    else:
        df = st.session_state.df

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Samples", len(df))
        col2.metric("Features", len(df.columns))
        col3.metric("Missing Values", int(df.isnull().sum().sum()))
        if "Potability" in df.columns:
            safe_pct = (df["Potability"] == 1).mean() * 100
            col4.metric("Safe (actual)", f"{safe_pct:.1f}%")

        st.subheader("📋 Statistical Summary")
        st.dataframe(df.describe().round(2), use_container_width=True)

        st.subheader("🔍 Missing Values")
        missing_df = pd.DataFrame({
            "Column": df.columns,
            "Missing": df.isnull().sum().values,
            "% Missing": (df.isnull().sum().values / len(df) * 100).round(2)
        })
        st.dataframe(missing_df, use_container_width=True)


# ==================== TAB 3: Predictions ====================
with tab3:
    st.subheader("🔮 Water Safety Predictions")

    if st.session_state.df is None:
        st.info("👈 Upload data first")
    elif st.session_state.model is None:
        st.warning("⚠️ No model loaded. Train one or load saved model from sidebar.")
    else:
        df = st.session_state.df

        if st.button("🚀 Run Predictions on All Rows", type="primary"):
            with st.spinner("Predicting..."):
                try:
                    # Drop target if present (prediction on features only)
                    features_df = df.drop(columns=["Potability"]) if "Potability" in df.columns else df
                    predictions = predict_from_dataframe(
                        features_df, st.session_state.model, st.session_state.scaler
                    )
                    st.session_state.predictions = predictions
                    st.success("✅ Predictions complete!")
                except Exception as e:
                    st.error(f"Prediction error: {e}")

        if st.session_state.predictions is not None:
            pred = st.session_state.predictions
            summary = get_prediction_summary(pred)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total", summary["total_samples"])
            c2.metric("Safe ✅", summary["safe_count"],
                      f"{summary['safe_percentage']}%")
            c3.metric("Unsafe ⚠️", summary["unsafe_count"],
                      f"{summary['unsafe_percentage']}%")
            c4.metric("Avg Confidence", f"{summary['avg_safety_probability']:.2%}")

            st.subheader("Detailed Predictions")
            st.dataframe(pred, use_container_width=True)

            csv_download = pred.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Predictions CSV",
                csv_download,
                "water_predictions.csv",
                "text/csv"
            )

        st.divider()
        st.subheader("🧪 Test a Single Sample")
        with st.form("single_sample"):
            cols = st.columns(3)
            sample = {}
            defaults = {
                "ph": 7.0, "Hardness": 200.0, "Solids": 22000.0,
                "Chloramines": 7.0, "Sulfate": 330.0, "Conductivity": 425.0,
                "Organic_carbon": 14.0, "Trihalomethanes": 66.0, "Turbidity": 4.0
            }
            for i, (k, v) in enumerate(defaults.items()):
                sample[k] = cols[i % 3].number_input(k, value=v, format="%.2f")

            submitted = st.form_submit_button("Predict This Sample")
            if submitted:
                try:
                    result = predict_single_sample(
                        sample, st.session_state.model, st.session_state.scaler
                    )
                    if result["Prediction"] == 1:
                        st.success(f"✅ SAFE - {result['Probability_Safe']:.1%} confidence")
                    else:
                        st.error(f"⚠️ UNSAFE - {(1-result['Probability_Safe']):.1%} confidence")
                except Exception as e:
                    st.error(f"Error: {e}")


# ==================== TAB 4: Visualizations ====================
with tab4:
    st.subheader("📈 Data Visualizations")

    if st.session_state.df is None:
        st.info("👈 Upload data first")
    else:
        df = handle_missing_values(st.session_state.df)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        viz_type = st.selectbox(
            "Select visualization",
            ["Distribution", "Correlation Heatmap", "Box Plot by Potability", "Scatter Plot"]
        )

        if viz_type == "Distribution":
            col = st.selectbox("Select column", numeric_cols)
            fig = px.histogram(df, x=col, nbins=40, color_discrete_sequence=["#0077b6"])
            fig.update_layout(title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Correlation Heatmap":
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                            aspect="auto", title="Feature Correlations")
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Box Plot by Potability":
            if "Potability" in df.columns:
                col = st.selectbox("Select column", [c for c in numeric_cols if c != "Potability"])
                fig = px.box(df, x="Potability", y=col, color="Potability",
                             color_discrete_map={0: "#ef4444", 1: "#10b981"})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Potability column not found")

        elif viz_type == "Scatter Plot":
            c1, c2 = st.columns(2)
            x_col = c1.selectbox("X axis", numeric_cols, index=0)
            y_col = c2.selectbox("Y axis", numeric_cols, index=1)
            color_col = "Potability" if "Potability" in df.columns else None
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                             opacity=0.6, title=f"{x_col} vs {y_col}")
            st.plotly_chart(fig, use_container_width=True)


# ==================== TAB 5: AI Chat ====================
with tab5:
    st.subheader("💬 Ask AI About Your Data")
    st.caption("Powered by LangChain + pandas dataframe agent")

    if st.session_state.df is None:
        st.info("👈 Upload data first")
    else:
        # Initialize agent
        if st.button("🤖 Initialize AI Agent"):
            with st.spinner("Setting up AI agent..."):
                provider = "groq" if llm_provider == "groq" else ("openai" if llm_provider == "openai" else "fallback")
                try:
                    agent = create_agent(
                        st.session_state.df,
                        provider=provider,
                        api_key=api_key
                    )
                    st.session_state.agent = agent
                    st.success("✅ Agent ready!")
                except Exception as e:
                    st.error(f"Agent error: {e}")

        if st.session_state.agent is not None:
            # Example questions
            st.caption("**Example questions:**")
            examples = [
                "What is the average pH?",
                "How many samples are unsafe?",
                "Which column has the most missing values?",
                "What is the correlation between pH and Potability?",
            ]
            cols = st.columns(len(examples))
            for i, ex in enumerate(examples):
                if cols[i].button(ex, key=f"ex_{i}"):
                    st.session_state._auto_q = ex

            # Chat interface
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            user_q = st.chat_input("Ask about your water quality data...")
            if hasattr(st.session_state, "_auto_q"):
                user_q = st.session_state._auto_q
                del st.session_state._auto_q

            if user_q:
                st.session_state.chat_history.append({"role": "user", "content": user_q})
                with st.chat_message("user"):
                    st.markdown(user_q)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        answer = st.session_state.agent.ask(user_q)
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})


# ==================== Footer ====================
st.divider()
st.caption("Built with Streamlit + scikit-learn + LangChain | Water Quality Detection AI v1.0")
