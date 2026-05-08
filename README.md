````md id="stu1"
# 📊 Student Marks Analysis & Prediction (Streamlit App)

A Streamlit-based data science project for analyzing student performance and predicting marks using machine learning models. It combines assignment, quiz, and exam data with weighted scoring and provides interactive visualizations and predictions.

---

## 🚀 Features

- 📊 Data analysis (EDA)
  - Histograms
  - Boxplots
  - Correlation heatmap
- 🧮 Weighted score calculation (Assignments, Quizzes, Exams)
- 🤖 Machine Learning models:
  - Linear Regression
  - Polynomial Regression
  - Dummy Model (baseline)
- 🔮 Interactive prediction system
- 📁 Excel & CSV dataset support
- 📈 Model comparison dashboard

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
- OpenPyXL

---

## 📁 Project Structure

```bash
project/
│
├── app.py                     # Streamlit main app
├── eda.py                    # Exploratory Data Analysis
├── model.py                  # ML model training & comparison
├── preprocess.py             # Data cleaning & preprocessing
├── marks_dataset.xlsx        # Raw dataset
├── preprocessed_dataset.csv  # Cleaned dataset
├── requirements.txt          # Dependencies
└── README.md
````

---

## ⚙️ How to Run

### 1️⃣ Clone repository

```bash
git clone <repository-url>
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit app

```bash
streamlit run app.py
```

---

## 📊 Workflow

1. Load dataset (Excel/CSV)
2. Preprocess data
3. Perform EDA visualizations
4. Train regression models
5. Compare model performance
6. Predict student marks using user input

---

## 🎯 Future Improvements

* Deep Learning model integration
* Student performance classification (Pass/Fail)
* Cloud deployment (Streamlit Cloud / AWS)
* Real-time database integration
* Student recommendation system

---

## 👨‍💻 Author

* Zeshan Haider

---

## 📄 License

This project is for educational and academic purposes.

```
```
