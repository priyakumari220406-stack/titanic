import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Titanic EDA Dashboard", layout="wide")

# Title
st.title("🚢 Titanic Data Analytics Dashboard")

# Load data
df = pd.read_csv("cleaned_titanic.csv")

# -------------------------------------
# Filters on main page (not sidebar)
# -------------------------------------
st.markdown("### 🔍 Apply Filters")

col_filter1, col_filter2, col_filter3, col_filter4 = st.columns(4)

with col_filter1:
    selected_genders = st.multiselect("Select Gender", options=df["Sex"].unique(), default=df["Sex"].unique())

with col_filter2:
    selected_pclass = st.multiselect("Select Passenger Class", options=sorted(df["Pclass"].unique()), default=sorted(df["Pclass"].unique()))

with col_filter3:
    min_age = int(df["Age"].min())
    max_age = int(df["Age"].max())
    selected_age_range = st.slider("Select Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))

with col_filter4:
    min_fare = int(df["Fare"].min())
    max_fare = int(df["Fare"].max())
    selected_fare_range = st.slider("Select Fare Range", min_value=min_fare, max_value=max_fare, value=(min_fare, max_fare))

# Filtered Data
filtered_df = df[
    (df["Sex"].isin(selected_genders)) &
    (df["Pclass"].isin(selected_pclass)) &
    (df["Age"].between(*selected_age_range)) &
    (df["Fare"].between(*selected_fare_range))
]

st.markdown("### 🎯 Filtered Data Preview")
st.dataframe(filtered_df.head(10))

# -------------------------------------
# Row 1: Survival Count & Age Distribution
# -------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📊 Survival Count by Gender")
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    sns.countplot(data=filtered_df, x="Survived", hue="Sex", palette="Set2", ax=ax1)
    ax1.set_title("Survival by Gender")
    st.pyplot(fig1)

with col2:
    st.markdown("#### 🎂 Age Distribution")
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    sns.histplot(filtered_df["Age"].dropna(), kde=True, bins=25, color="orange", ax=ax2)
    ax2.set_title("Age Distribution")
    st.pyplot(fig2)

# -------------------------------------
# Row 2: Survival by Class & Fare Distribution
# -------------------------------------
col3, col4 = st.columns(2)

with col3:
    st.markdown("#### 🏷️ Survival Rate by Class")
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    sns.barplot(data=filtered_df, x="Pclass", y="Survived", ci=None, palette="Blues", ax=ax3)
    ax3.set_title("Survival by Class")
    st.pyplot(fig3)

with col4:
    st.markdown("#### 💰 Fare Distribution")
    fig4, ax4 = plt.subplots(figsize=(5, 3))
    sns.histplot(filtered_df["Fare"].dropna(), kde=True, bins=30, color="gold", ax=ax4)
    ax4.set_title("Fare Distribution")
    st.pyplot(fig4)

# -------------------------------------
# Row 3: Average Fare by Gender & Average Age by Class
# -------------------------------------
col5, col6 = st.columns(2)

with col5:
    st.markdown("#### 📌 Average Fare by Gender")
    fig5, ax5 = plt.subplots(figsize=(5, 3))
    sns.barplot(data=filtered_df, x="Sex", y="Fare", palette="pastel", ax=ax5)
    ax5.set_title("Average Fare by Gender")
    st.pyplot(fig5)

with col6:
    st.markdown("#### 🎓 Average Age by Class")
    fig6, ax6 = plt.subplots(figsize=(5, 3))
    sns.barplot(data=filtered_df, x="Pclass", y="Age", palette="Purples", ax=ax6)
    ax6.set_title("Average Age by Class")
    st.pyplot(fig6)

# -------------------------------------
# Row 4: Embarkation Count & Survival Rate by Gender
# -------------------------------------
col7, col8 = st.columns(2)

with col7:
    st.markdown("#### 🧾 Passenger Count by Embarkation")
    fig7, ax7 = plt.subplots(figsize=(5, 3))
    sns.countplot(data=filtered_df, x="Embarked", palette="Set3", ax=ax7)
    ax7.set_title("Embarkation Port Count")
    st.pyplot(fig7)

with col8:
    st.markdown("#### ✅ Survival Rate by Gender")
    fig8, ax8 = plt.subplots(figsize=(5, 3))
    sns.barplot(data=filtered_df, x="Sex", y="Survived", ci=None, palette="coolwarm", ax=ax8)
    ax8.set_title("Survival Rate by Gender")
    st.pyplot(fig8)
