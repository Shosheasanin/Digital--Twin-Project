# 💧 Water Quality Detection AI

A full-featured AI app that analyzes water quality data, predicts safety, and answers natural language questions using LangChain.

## Features

- 📁 **CSV Upload** — Drop any water quality dataset
- 🤖 **ML Predictions** — Random Forest classifier (potable vs non-potable)
- 📊 **Interactive Visualizations** — Plotly charts, heatmaps, box plots
- 💬 **LangChain AI Chat** — Ask questions in plain English
- 🧪 **Single Sample Testing** — Test one water sample manually
- ⬇️ **Export Results** — Download predictions as CSV

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset
Get the water potability dataset from Kaggle:
- https://www.kaggle.com/datasets/adityakadiwal/water-potability
- Save as `data/water_potability.csv`

### 3. Get a free API key (for chat)
- **Groq** (recommended, free): https://console.groq.com
- **OpenAI** (paid): https://platform.openai.com

### 4. Run the app
```bash
streamlit run app.py
```

## Usage Flow

1. **Upload Data** tab → upload your CSV or click "Load Sample Data"
2. Sidebar → **Train New Model** (or Load Saved Model)
3. **Predictions** tab → Run predictions on all rows
4. **Visualizations** tab → Explore your data
5. **AI Chat** tab → Enter API key, initialize agent, ask questions

## Project Structure

```
water-quality-ai/
├── app.py                    # Streamlit frontend
├── requirements.txt
├── data/                     # Your CSV files go here
├── models/                   # Saved ML models
└── src/
    ├── preprocess.py         # Data cleaning & scaling
    ├── train_model.py        # Random Forest training
    ├── predict.py            # Prediction logic
    └── langchain_agent.py    # LangChain Q&A agent
```

## Expected CSV Columns

`ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity, Potability`

## Tech Stack

- **ML**: scikit-learn Random Forest
- **AI Chat**: LangChain + Groq/OpenAI
- **UI**: Streamlit
- **Viz**: Plotly

## Example Questions for AI Chat

- "What is the average pH of safe water?"
- "How many samples have turbidity above 5?"
- "Which feature correlates most with potability?"
- "Show me unsafe samples with high chloramines"
