import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# ----------------------------
# Function to set background
# ----------------------------
def set_background(png_file):
    with open(png_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ----------------------------
# Set up the page
# ----------------------------
st.set_page_config(page_title="Titanic EDA Dashboard", layout="wide")
set_background("b6084c84-d281-4fc0-9e3d-3f313decf481.png")

# Title
st.title("🚢 Titanic Data Analytics Dashboard")

# Load data
df = pd.read_csv("cleaned_titanic.csv")

# Filters on main page
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

# ---------------------
# Visualizations
# ---------------------
def plot_chart(fig, title):
    st.markdown(f"#### {title}")
    st.pyplot(fig)

# Row 1
col1, col2 = st.columns(2)
with col1:
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    sns.countplot(data=filtered_df, x="Survived", hue="Sex", palette="Set2", ax=ax1)
    ax1.set_title("Survival by Gender")
    plot_chart(fig1, "📊 Survival Count by Gender")

with col2:
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    sns.histplot(filtered_df["Age"].dropna(), kde=True, bins=25, color="orange", ax=ax2)
    ax2.set_title("Age Distribution")
    plot_chart(fig2, "🎂 Age Distribution")

# Row 2
col3, col4 = st.columns(2)
with col3:
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    sns.barplot(data=filtered_df, x="Pclass", y="Survived", ci=None, palette="Blues", ax=ax3)
    ax3.set_title("Survival by Class")
    plot_chart(fig3, "🏷️ Survival Rate by Class")

with col4:
    fig4, ax4 = plt.subplots(figsize=(5, 3))
    sns.histplot(filtered_df["Fare"].dropna(), kde=True, bins=30, color="gold", ax=ax4)
    ax4.set_title("Fare Distribution")
    plot_chart(fig4, "💰 Fare Distribution")

# Row 3
col5, col6 = st.columns(2)
with col5:
    fig5, ax5 = plt.subplots(figsize=(5, 3))
    sns.barplot(data=filtered_df, x="Sex", y="Fare", palette="pastel", ax=ax5)
    ax5.set_title("Average Fare by Gender")
    plot_chart(fig5, "📌 Average Fare by Gender")

with col6:
    fig6, ax6 = plt.subplots(figsize=(5, 3))
    sns.barplot(data=filtered_df, x="Pclass", y="Age", palette="Purples", ax=ax6)
    ax6.set_title("Average Age by Class")
    plot_chart(fig6, "🎓 Average Age by Class")

# Row 4
col7, col8 = st.columns(2)
with col7:
    fig7, ax7 = plt.subplots(figsize=(5, 3))
    sns.countplot(data=filtered_df, x="Embarked", palette="Set3", ax=ax7)
    ax7.set_title("Embarkation Port Count")
    plot_chart(fig7, "🧾 Passenger Count by Embarkation")

with col8:
    fig8, ax8 = plt.subplots(figsize=(5, 3))
    sns.barplot(data=filtered_df, x="Sex", y="Survived", ci=None, palette="coolwarm", ax=ax8)
    ax8.set_title("Survival Rate by Gender")
    plot_chart(fig8, "✅ Survival Rate by Gender")
