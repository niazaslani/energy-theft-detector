# Energy Theft Detector

This is a machine learning web application that detects potential energy theft from household electricity usage data.

## 🔍 Features
- Upload your `.csv` dataset
- View live predictions of suspected energy theft
- See tabular outputs with a clean "Theft Prediction" column

## 🚀 Live App
[https://energy-theft-detector.streamlit.app/](https://energy-theft-detector.streamlit.app/)

## 🗂️ File Structure
```
project_root/
├── app.py
├── rf_model.pkl
├── new_profiles.csv
├── predictions.csv
├── step1_select_users.py
├── step1_train_model.py
├── step6_test_model.py
├── requirements.txt
├── .gitignore
└── sample_with_thefts.csv
```

## ⚙️ Setup & Run Locally

1. Clone the repository
```bash
git clone https://github.com/niazaslani/energy-theft-detector.git
cd energy-theft-detector
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the app
```bash
streamlit run app.py
```

## 📌 Requirements
- Python 3.8+
- Streamlit
- pandas
- scikit-learn
- joblib
- matplotlib
- seaborn

## 🧠 Model Info
The model was trained using a Random Forest classifier on energy consumption data with synthetic balancing for theft vs non-theft records.

## 📄 License
This project is open source and available under the [MIT License](LICENSE).