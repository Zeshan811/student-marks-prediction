import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_histogram(df, col):
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, bins=15, ax=ax)
    ax.set_title(f"Histogram of {col}")
    st.pyplot(fig)

def plot_boxplot(df, col):
    fig, ax = plt.subplots()
    sns.boxplot(x=df[col], ax=ax)
    ax.set_title(f"Boxplot of {col}")
    st.pyplot(fig)

def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
