import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🧪 Data Validation Tool")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Preview of Data")
    st.dataframe(df.head())

    if st.button("Run Validation Checks"):
        st.subheader("✅ Validation Results")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        st.write("Data Types:")
        st.write(df.dtypes)

        st.subheader("❌ Missing Values")
        st.write(df.isnull().sum())

        st.subheader("📎 Duplicate Rows")
        st.write(f"Duplicate Rows: {df.duplicated().sum()}")

        st.subheader("📉 Data Distribution")
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

        st.subheader("🔗 Correlation Heatmap")
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots()
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.write("Not enough numeric columns for correlation heatmap.")
