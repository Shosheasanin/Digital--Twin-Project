import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


st.set_page_config(
    page_title="Water Quality Predictor",
    layout="wide"
)
st.info("""
This project uses water quality parameters such as pH, hardness, solids, chloramines,
sulfate, conductivity, organic carbon, trihalomethanes and turbidity to predict
whether water is safe or unsafe for drinking.
""")
st.title(" Water Quality Predictor")
st.write("This app predicts whether water is safe or unsafe using Machine Learning.")


# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("water.csv")
    return df


try:
    df = load_data()
except:
    st.error("water.csv file not found. Please place water.csv in the same folder as app.py")
    st.stop()


# Fill missing values
df = df.fillna(df.mean(numeric_only=True))


st.subheader(" Dataset Preview")
st.dataframe(df.head())


# Check required column
if "Potability" not in df.columns:
    st.error("Dataset must contain 'Potability' column.")
    st.stop()


# Split data
X = df.drop("Potability", axis=1)
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


st.subheader(" Model Performance")
st.success(f"Model Accuracy: {accuracy * 100:.2f}%")


# Charts
st.subheader(" Water Safety Distribution")

count_df = df["Potability"].value_counts().reset_index()
count_df.columns = ["Potability", "Count"]
count_df["Potability"] = count_df["Potability"].replace({
    0: "Unsafe",
    1: "Safe"
})

fig = px.pie(
    count_df,
    names="Potability",
    values="Count",
    title="Safe vs Unsafe Water Samples"
)

st.plotly_chart(fig, use_container_width=True)


st.subheader(" Test a Water Sample")

input_data = {}

for col in X.columns:
    input_data[col] = st.number_input(
        f"Enter {col}",
        value=float(df[col].mean())
    )


if st.button("Predict Water Quality"):
    sample = pd.DataFrame([input_data])
    prediction = model.predict(sample)[0]

    if prediction == 1:
        st.success(" The water is predicted to be SAFE.")
    else:
        st.error(" The water is predicted to be UNSAFE.")


st.subheader("⬇ Download Dataset")

csv = df.to_csv(index=False)

st.download_button(
    label="Download Cleaned Dataset",
    data=csv,
    file_name="cleaned_water_data.csv",
    mime="text/csv"
)