import streamlit as st
import pandas as pd
from preprocess import preprocess_dataset
from model import evaluate_models, create_comparison_table
from eda import plot_histogram, plot_boxplot, plot_correlation_heatmap
import matplotlib.pyplot as plt

# ------------------ Load & Preprocess Dataset ------------------
st.title("Student Marks Analysis & Prediction")
df = preprocess_dataset("marks_dataset.xlsx")

# Sidebar main options
st.sidebar.header("Main Options")
main_option = st.sidebar.radio("Select Option:", ["EDA", "Prediction", "Comparison Table"])

# ------------------ EDA ------------------
if main_option == "EDA":
    st.header("Exploratory Data Analysis")
    eda_choice = st.selectbox("Choose EDA type:", ["Histogram", "Boxplot", "Correlation Heatmap"])
    cols = ['Assignments', 'Quizzes', 'Mid1', 'Mid2', 'Final']

    if eda_choice in ["Histogram", "Boxplot"]:
        selected_col = st.selectbox("Select Column:", cols)
        if eda_choice == "Histogram":
            plot_histogram(df, selected_col)
        elif eda_choice == "Boxplot":
            plot_boxplot(df, selected_col)
    elif eda_choice == "Correlation Heatmap":
        plot_correlation_heatmap(df)

# ------------------ Prediction ------------------
elif main_option == "Prediction":
    st.header("Predict Student Marks")
    pred_choice = st.selectbox("Select Target:", ["Mid1", "Mid2", "Final"])

    # Prepare feature inputs based on target
    if pred_choice == "Mid1":
        features = ['Assignments','Quizzes']
        X, y = df[features], df['Mid1']
    elif pred_choice == "Mid2":
        features = ['Assignments','Quizzes','Mid1']
        X, y = df[features], df['Mid2']
    else:  # Final
        features = ['Assignments','Quizzes','Mid1','Mid2']
        X, y = df[features], df['Final']

    # Train models
    results, models = evaluate_models(X, y, poly_degree=2)

    st.subheader("Enter Feature Values for Prediction")
    user_input = {}
    for feat in features:
       if feat == "Assignments":
         user_input[feat] = st.number_input(f"{feat} (0-15):", min_value=0.0, max_value=15.0, value=0.0)
       elif feat == "Quizzes":
         user_input[feat] = st.number_input(f"{feat} (0-10):", min_value=0.0, max_value=10.0, value=0.0)
       elif feat in ["Mid1","Mid2"]:
         user_input[feat] = st.number_input(f"{feat} (0-15):", min_value=0.0, max_value=15.0, value=0.0)

    # Dropdown to select model
    selected_model = st.selectbox("Select Model for Prediction:", list(models.keys()))

    if st.button("Predict"):
        input_df = pd.DataFrame([user_input])
        model_info = models[selected_model]
        from model import predict_marks
        pred = predict_marks(model_info, input_df)
        st.success(f"{selected_model} Prediction: {pred[0]:.2f}")

# ------------------ Comparison Table ------------------
elif main_option == "Comparison Table":
    st.header("Model Comparison Table")
    target_choice = st.selectbox("Select Target:", ["Mid1", "Mid2", "Final"])

    # Prepare feature inputs based on target
    if target_choice == "Mid1":
        X, y = df[['Assignments','Quizzes']], df['Mid1']
    elif target_choice == "Mid2":
        X, y = df[['Assignments','Quizzes','Mid1']], df['Mid2']
    else:
        X, y = df[['Assignments','Quizzes','Mid1','Mid2']], df['Final']

    # Train models
    results, models = evaluate_models(X, y, poly_degree=2)
    st.dataframe(create_comparison_table(results))

    # Optional Train vs Test R2 graph
    if st.checkbox("Show Train vs Test R² Graph"):
        train_r2 = [v['Train_R2'] for v in results.values()]
        test_r2 = [v['R2'] for v in results.values()]
        model_names = list(results.keys())

        fig, ax = plt.subplots()
        ax.bar([x-0.2 for x in range(len(model_names))], train_r2, width=0.4, label="Train R²")
        ax.bar([x+0.2 for x in range(len(model_names))], test_r2, width=0.4, label="Test R²")
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names)
        ax.set_ylabel("R² Score")
        ax.set_title(f"Train vs Test R² for {target_choice}")
        ax.legend()
        st.pyplot(fig)
