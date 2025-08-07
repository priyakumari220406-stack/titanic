import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")

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

# Gender filter
gender = st.sidebar.multiselect("Select Gender", options=df["Sex"].unique(), default=list(df["Sex"].unique()))

# Pclass filter
pclass = st.sidebar.multiselect("Select Passenger Class", options=sorted(df["Pclass"].unique()), default=sorted(df["Pclass"].unique()))

# Age slider
min_age = int(df["Age"].min())
max_age = int(df["Age"].max())
age_range = st.sidebar.slider("Select Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))

# Fare slider
min_fare = int(df["Fare"].min())
max_fare = int(df["Fare"].max())
fare_range = st.sidebar.slider("Select Fare Range", min_value=min_fare, max_value=max_fare, value=(min_fare, max_fare))

# Filter Data
filtered_df = df[
    (df["Sex"].isin(gender)) &
    (df["Pclass"].isin(pclass)) &
    (df["Age"].between(age_range[0], age_range[1])) &
    (df["Fare"].between(fare_range[0], fare_range[1]))
]

st.subheader("🎯 Filtered Data Preview")
st.write(filtered_df.head())

# ================================
# 🔸 Row 1: Survival Count & Age Distribution
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📊 Survival Count by Gender")
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    sns.countplot(data=filtered_df, x="Survived", hue="Sex", ax=ax1, palette="pastel")
    ax1.set_title("Survival by Gender")
    st.pyplot(fig1)

with col2:
    st.markdown("#### 🎂 Age Distribution")
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    sns.histplot(filtered_df["Age"].dropna(), kde=True, bins=25, ax=ax2, color='skyblue')
    ax2.set_title("Age Distribution")
    st.pyplot(fig2)

# ================================
# 🔸 Row 2: Survival by Class & Fare Distribution
col3, col4 = st.columns(2)

with col3:
    st.markdown("#### 🏷️ Survival Rate by Class")
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    sns.barplot(data=filtered_df, x="Pclass", y="Survived", ci=None, ax=ax3, palette="Blues")
    ax3.set_title("Survival by Class")
    st.pyplot(fig3)

with col4:
    st.markdown("#### 💰 Fare Distribution")
    fig4, ax4 = plt.subplots(figsize=(5, 3))
    sns.histplot(filtered_df["Fare"], kde=True, bins=30, ax=ax4, color='orange')
    ax4.set_title("Fare Distribution")
    st.pyplot(fig4)

# ================================
# 🔸 Row 3: Avg Fare by Gender & Avg Age by Class
col5, col6 = st.columns(2)

with col5:
    st.markdown("#### 📌 Average Fare by Gender")
    fig5, ax5 = plt.subplots(figsize=(5, 3))
    sns.barplot(data=filtered_df, x="Sex", y="Fare", ax=ax5, palette="Set2")
    ax5.set_title("Average Fare by Gender")
    st.pyplot(fig5)

with col6:
    st.markdown("#### 🎓 Average Age by Class")
    fig6, ax6 = plt.subplots(figsize=(5, 3))
    sns.barplot(data=filtered_df, x="Pclass", y="Age", ax=ax6, palette="Set3")
    ax6.set_title("Average Age by Class")
    st.pyplot(fig6)

# ================================
# 🔸 Row 4: Embark Count & Survival by Gender
col7, col8 = st.columns(2)

with col7:
    st.markdown("#### 🧾 Passenger Count by Embarkation")
    fig7, ax7 = plt.subplots(figsize=(5, 3))
    sns.countplot(data=filtered_df, x="Embarked", ax=ax7, palette="pastel")
    ax7.set_title("Embarkation Port Count")
    st.pyplot(fig7)

with col8:
    st.markdown("#### ✅ Survival Rate by Gender")
    fig8, ax8 = plt.subplots(figsize=(5, 3))
    sns.barplot(data=filtered_df, x="Sex", y="Survived", ci=None, ax=ax8, palette="coolwarm")
    ax8.set_title("Survival Rate by Gender")
    st.pyplot(fig8)

# ================================
# 🔸 Row 5: Correlation Heatmap & Pie Chart
col9, col10 = st.columns(2)

with col9:
    st.markdown("#### 🧠 Correlation Heatmap")
    fig9, ax9 = plt.subplots(figsize=(6, 4))
    corr = filtered_df.select_dtypes(include=['float64', 'int64']).corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax9)
    ax9.set_title("Feature Correlation")
    st.pyplot(fig9)

with col10:
    st.markdown("#### 🧩 Survival Pie Chart")
    survived_counts = filtered_df["Survived"].value_counts()
    fig10, ax10 = plt.subplots(figsize=(5, 3))
    colors = ['salmon', 'lightgreen']
    ax10.pie(survived_counts, labels=["Not Survived", "Survived"],
             autopct='%1.1f%%', startangle=90, colors=colors)
    ax10.set_title("Survival Distribution")
    ax10.axis("equal")
    st.pyplot(fig10)

# ================================
# 🔸 Full-width: Survival by Embarkation Port
st.markdown("#### 🛳️ Survival by Embarkation Port")
fig11, ax11 = plt.subplots(figsize=(10, 3))
sns.countplot(data=filtered_df, x="Embarked", hue="Survived", ax=ax11, palette="muted")
ax11.set_title("Survival by Embarkation Port")
st.pyplot(fig11)

# Footer
st.markdown("---")
st.markdown("💡 *Use the sidebar to explore Titanic passengers by class, gender, age, and fare.*")
