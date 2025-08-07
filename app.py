import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="Titanic EDA Dashboard", layout="wide")

# Load Data
df = pd.read_csv("cleaned_titanic.csv")

# Title
st.title("🚢 Titanic Data Analytics Dashboard")

# Show Raw Data
if st.checkbox("Show Raw Data"):
    st.dataframe(df)

# Sidebar Filters
st.sidebar.header("🔍 Filter Options")
gender = st.sidebar.selectbox("Select Gender", options=df["Sex"].unique())
pclass = st.sidebar.selectbox("Select Passenger Class", options=sorted(df["Pclass"].unique()))

# Filter Data
filtered_df = df[(df["Sex"] == gender) & (df["Pclass"] == pclass)]

st.subheader("🎯 Filtered Data Preview")
st.write(filtered_df.head())

# ========== 🔹 VISUAL SECTION 1 ==========

st.markdown("## 📊 Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Survival Count by Gender")
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    sns.countplot(data=filtered_df, x="Survived", hue="Sex", ax=ax1)
    ax1.set_title("Survived vs Not Survived")
    st.pyplot(fig1)

with col2:
    st.markdown("#### Age Distribution")
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    sns.histplot(filtered_df["Age"].dropna(), kde=True, bins=30, ax=ax2)
    ax2.set_title("Age Distribution")
    st.pyplot(fig2)

# ========== 🔹 VISUAL SECTION 2 ==========

col3, col4 = st.columns(2)

with col3:
    st.markdown("#### Survival Rate by Class")
    fig3, ax3 = p
