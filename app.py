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

# Show Data
if st.checkbox("Show Raw Data"):
    st.dataframe(df)

# Sidebar Filters
st.sidebar.header("Filter Options")
gender = st.sidebar.selectbox("Select Gender", options=df["Sex"].unique())
pclass = st.sidebar.selectbox("Select Passenger Class", options=sorted(df["Pclass"].unique()))

# Apply filters
filtered_df = df[(df["Sex"] == gender) & (df["Pclass"] == pclass)]

st.subheader("🎯 Filtered Data Preview")
st.write(filtered_df.head())

# =======================
# 🔹 Visualization 1: Survival Count by Gender
st.subheader("📊 Survival Count by Gender")
fig1, ax1 = plt.subplots()
sns.countplot(data=filtered_df, x="Survived", hue="Sex", ax=ax1)
ax1.set_title("Survived vs Not Survived by Gender")
st.pyplot(fig1)

# 🔹 Visualization 2: Age Distribution
st.subheader("🎂 Age Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(filtered_df["Age"].dropna(), kde=True, bins=30, ax=ax2)
ax2.set_title("Age Distribution of Passengers")
st.pyplot(fig2)

# 🔹 Visualization 3: Survival Rate by Class
st.subheader("🏷️ Survival Rate by Passenger Class")
fig3, ax3 = plt.subplots()
sns.barplot(data=df, x="Pclass", y="Survived", ci=None, ax=ax3)
ax3.set_title("Survival Rate by Passenger Class")
st.pyplot(fig3)

# 🔹 Visualization 4: Fare Distribution
st.subheader("💰 Fare Distribution")
fig4, ax4 = plt.subplots()
sns.histplot(df["Fare"], kde=True, bins=40, ax=ax4)
ax4.set_title("Fare Distribution")
st.pyplot(fig4)

# 🔹 Visualization 5: Correlation Heatmap
st.subheader("🧠 Correlation Heatmap")
fig5, ax5 = plt.subplots(figsize=(10, 6))
corr = df.select_dtypes(include=['float64', 'int64']).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax5)
ax5.set_title("Correlation Between Features")
st.pyplot(fig5)

# 🔹 Visualization 6: Pie Chart of Survival
st.subheader("🧩 Survival Distribution")
survived_counts = df["Survived"].value_counts()
fig6, ax6 = plt.subplots()
ax6.pie(survived_counts, labels=["Not Survived", "Survived"], autopct='%1.1f%%', startangle=90)
ax6.set_title("Overall Survival Rate")
ax6.axis("equal")
st.pyplot(fig6)

# 🔹 Visualization 7: Embarked Distribution
st.subheader("🛳️ Passenger Embarkation Port")
fig7, ax7 = plt.subplots()
sns.countplot(data=df, x="Embarked", hue="Survived", ax=ax7)
ax7.set_title("Survival by Embarkation Port")
st.pyplot(fig7)

# Footer
st.markdown("---")
st.markdown("💡 *Explore the Titanic dataset by filtering and visualizing various insights.*")
