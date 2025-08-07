import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="Titanic EDA Dashboard", layout="wide")

# Title
st.title("🚢 Titanic Data Analytics Dashboard")

# Load Data
df = pd.read_csv("cleaned_titanic.csv")

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

# ================================
# 🔸 Row 1: Survival Count & Age Distribution
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📊 Survival Count by Gender")
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    sns.countplot(data=filtered_df, x="Survived", hue="Sex", ax=ax1)
    ax1.set_title("Survival by Gender")
    st.pyplot(fig1)

with col2:
    st.markdown("#### 🎂 Age Distribution")
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    sns.histplot(filtered_df["Age"].dropna(), kde=True, bins=25, ax=ax2)
    ax2.set_title("Age Distribution")
    st.pyplot(fig2)

# ================================
# 🔸 Row 2: Survival by Class & Fare
col3, col4 = st.columns(2)

with col3:
    st.markdown("#### 🏷️ Survival Rate by Class")
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    sns.barplot(data=df, x="Pclass", y="Survived", ci=None, ax=ax3)
    ax3.set_title("Survival by Class")
    st.pyplot(fig3)

with col4:
    st.markdown("#### 💰 Fare Distribution")
    fig4, ax4 = plt.subplots(figsize=(5,3))
    # ================================
# 🔸 Row 4: Avg Fare by Gender & Avg Age by Class
col7, col8 = st.columns(2)

with col7:
    st.markdown("#### 📌 Average Fare by Gender")
    fig8, ax8 = plt.subplots(figsize=(5, 3))
    sns.barplot(data=df, x="Sex", y="Fare", ax=ax8)
    ax8.set_title("Average Fare by Gender")
    st.pyplot(fig8)

with col8:
    st.markdown("#### 🎓 Average Age by Class")
    fig9, ax9 = plt.subplots(figsize=(5, 3))
    sns.barplot(data=df, x="Pclass", y="Age", ax=ax9)
    ax9.set_title("Average Age by Class")
    st.pyplot(fig9)

# ================================
# 🔸 Row 5: Passenger Count by Embark & Survival by Gender
col9, col10 = st.columns(2)

with col9:
    st.markdown("#### 🧾 Passenger Count by Embarkation")
    fig10, ax10 = plt.subplots(figsize=(5, 3))
    sns.countplot(data=df, x="Embarked", ax=ax10)
    ax10.set_title("Embarkation Port Count")
    st.pyplot(fig10)

with col10:
    st.markdown("#### ✅ Survival Rate by Gender")
    fig11, ax11 = plt.subplots(figsize=(5, 3))
    sns.barplot(data=df, x="Sex", y="Survived", ci=None, ax=ax11)
    ax11.set_title("Survival Rate by Gender")
    st.pyplot(fig11)



