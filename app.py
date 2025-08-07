import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# Page Config
st.set_page_config(page_title="Titanic EDA Dashboard", layout="wide")

# ===== Background CSS =====
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1607746882042-944635dfe10e");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    .block-container {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 10px;
    }

    .css-1v0mbdj, .css-1dp5vir {
        background-color: rgba(255, 255, 255, 0.8) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===== Title =====
st.title("🚢 Titanic Data Analytics Dashboard")

# ===== Load Data =====
df = pd.read_csv("cleaned_titanic.csv")

# ===== Show Raw Data =====
if st.checkbox("Show Raw Data"):
    st.dataframe(df)

# ===== Sidebar Filters =====
st.sidebar.header("🔍 Filter Options")

gender = st.sidebar.multiselect("Select Gender", options=df["Sex"].unique(), default=list(df["Sex"].unique()))
pclass = st.sidebar.multiselect("Select Passenger Class", options=sorted(df["Pclass"].unique()), default=sorted(df["Pclass"].unique()))
age_range = st.sidebar.slider("Select Age Range", min_value=int(df["Age"].min()), max_value=int(df["Age"].max()), value=(int(df["Age"].min()), int(df["Age"].max())))
fare_range = st.sidebar.slider("Select Fare Range", min_value=int(df["Fare"].min()), max_value=int(df["Fare"].max()), value=(int(df["Fare"].min()), int(df["Fare"].max())))

# ===== Filter Data =====
filtered_df = df[
    (df["Sex"].isin(gender)) &
    (df["Pclass"].isin(pclass)) &
    (df["Age"].between(age_range[0], age_range[1])) &
    (df["Fare"].between(fare_range[0], fare_range[1]))
]

st.subheader("🎯 Filtered Data Preview")
st.write(filtered_df.head())

# ===== ROW 1 =====
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 📊 Survival Count by Gender")
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    sns.countplot(data=filtered_df, x="Survived", hue="Sex", ax=ax1, palette="pastel")
    st.pyplot(fig1)

with col2:
    st.markdown("#### 🎂 Age Distribution")
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    sns.histplot(filtered_df["Age"].dropna(), kde=True, bins=25, ax=ax2, color='skyblue')
    st.pyplot(fig2)

# ===== ROW 2 =====
col3, col4 = st.columns(2)
with col3:
    st.markdown("#### 🏷️ Survival Rate by Class")
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    sns.barplot(data=filtered_df, x="Pclass", y="Survived", ci=None, ax=ax3, palette="Blues")
    st.pyplot(fig3)

with col4:
    st.markdown("#### 💰 Fare Distribution")
    fig4, ax4 = plt.subplots(figsize=(5, 3))
    sns.histplot(filtered_df["Fare"], kde=True, bins=30, ax=ax4, color='orange')
    st.pyplot(fig4)

# ===== ROW 3 =====
col5, col6 = st.columns(2)
with col5:
    st.markdown("#### 📌 Average Fare by Gender")
    fig5, ax5 = plt.subplots(figsize=(5, 3))
    sns.barplot(data=filtered_df, x="Sex", y="Fare", ax=ax5, palette="Set2")
    st.pyplot(fig5)

with col6:
    st.markdown("#### 🎓 Average Age by Class")
    fig6, ax6 = plt.subplots(figsize=(5, 3))
    sns.barplot(data=filtered_df, x="Pclass", y="Age", ax=ax6, palette="Set3")
    st.pyplot(fig6)

# ===== ROW 4 =====
col7, col8 = st.columns(2)
with col7:
    st.markdown("#### 🧾 Passenger Count by Embarkation")
    fig7, ax7 = plt.subplots(figsize=(5, 3))
    sns.countplot(data=filtered_df, x="Embarked", ax=ax7, palette="pastel")
    st.pyplot(fig7)

with col8:
    st.markdown("#### ✅ Survival Rate by Gender")
    fig8, ax8 = plt.subplots(figsize=(5, 3))
    sns.barplot(data=filtered_df, x="Sex", y="Survived", ci=None, ax=ax8, palette="coolwarm")
    st.pyplot(fig8)


