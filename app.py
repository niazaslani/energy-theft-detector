import streamlit as st
import pandas as pd
import joblib
import base64
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Energy Theft Detection", layout="centered")

st.title("🔌 Energy Theft Detection Web App")
st.markdown("""
This app uses a trained machine learning model to predict electricity theft
based on household energy usage patterns.

📁 **Upload a CSV file** with electricity usage profiles.
📊 **Review predictions** and download results.
🔒 *Private and secure – no data is stored.*
""")

st.sidebar.title("About")
st.sidebar.markdown("""
**Developer:** Niaz Aslani  
**App:** Energy Theft Detector  
**Model:** Random Forest (balanced)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
To use:
- CSV must include **LCLid** and **KWH/hh (per half hour)** columns.
- Max file size: 200 MB.
""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File successfully loaded!")

        if "KWH/hh (per half hour)" not in df.columns:
            st.error("Missing required column: 'KWH/hh (per half hour)'")
        else:
            st.subheader("📋 Uploaded Data")
            st.dataframe(df.head())

            drop_cols = ["LCLid", "DateTime"] if "DateTime" in df.columns else ["LCLid"]
            clean_data = df.drop(columns=drop_cols, errors="ignore")
            X = clean_data.select_dtypes(include="number")

            model = joblib.load("rf_model.pkl")
            df["Theft_Prediction"] = model.predict(X)

            st.subheader("📊 Prediction Results")
            st.dataframe(df[["LCLid", "KWH/hh (per half hour)", "Theft_Prediction"]])

            theft_count = df["Theft_Prediction"].value_counts()
            fig, ax = plt.subplots()
            sns.barplot(x=theft_count.index, y=theft_count.values, ax=ax)
            ax.set_xticklabels(["No Theft", "Theft"])
            ax.set_ylabel("Number of Users")
            ax.set_title("Theft Detection Summary")
            st.pyplot(fig)

            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode("utf-8")

            csv = convert_df_to_csv(df)
            b64 = base64.b64encode(csv).decode()  # for download button fallback
            href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📥 Download Predictions CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("Please upload a .csv file to begin.")