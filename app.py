import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Energy Theft Detection", layout="wide")
st.title("Energy Theft Detection Web App")
st.markdown("""
Upload a CSV file with electricity usage profiles (e.g., mean consumption) to detect theft. 
The model will analyze the data and predict whether theft is likely.
""")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.subheader("Uploaded Data")
        st.dataframe(data.head(), use_container_width=True)

        # Clean input
        drop_cols = ["LCLid", "DateTime"] if "DateTime" in data.columns else ["LCLid"]
        clean_data = data.drop(columns=drop_cols, errors="ignore")
        X = clean_data.select_dtypes(include="number")

        # Load model and predict
        rf = joblib.load("rf_model.pkl")
        preds = rf.predict(X)
        data["Theft_Prediction"] = preds

        # Show results
        st.subheader("Prediction Results")
        st.dataframe(data, use_container_width=True)

        # Show summary
        theft_counts = data["Theft_Prediction"].value_counts().rename({0: "Normal", 1: "Theft"})
        st.subheader("Summary")
        st.bar_chart(theft_counts)

        # Download button
        csv_output = data.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Results as CSV",
            data=csv_output,
            file_name="predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing the file: {e}")
else:
    st.info("Please upload a .csv file to begin.")