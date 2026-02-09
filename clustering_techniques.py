import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("Clustering Explorer with K-Means")

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("flight.csv")
    return df

df = load_data()

st.subheader("Raw Data")
st.dataframe(df.head())

# ================= PREPROCESSING =================
st.subheader("Preprocessing")

# Handle date to numeric
date_cols = ['FFP_DATE','FIRST_FLIGHT_DATE','LAST_FLIGHT_DATE','LOAD_TIME']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')
    df[col] = (df[col] - df[col].min()).dt.days

# Missing values numeric
num_cols = df.select_dtypes(include=['int64','float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Missing values categorical
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna("Unknown")

# Encoding gender
df['GENDER'] = df['GENDER'].map({'Male': 1, 'Female': 0}).fillna(-1)

# Drop irrelevant columns
drop_cols = ['MEMBER_NO', 'WORK_CITY', 'WORK_PROVINCE', 'WORK_COUNTRY']
df = df.drop(columns=drop_cols, errors='ignore')

st.write("Data after preprocessing")
st.dataframe(df.head())

# ================= SCALING =================
df_numeric = df.select_dtypes(include=["int64", "float64"])
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

# ================= SIDEBAR CONTROL =================
st.sidebar.header("K-Means Settings")
k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 4)

# ================= KMEANS =================
kmeans = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = kmeans.fit_predict(df_scaled)

# ================= RESULT =================
st.subheader("Cluster Distribution")
st.write(df["Cluster"].value_counts())

# ================= VISUALIZATION =================
st.subheader("Visualization: FLIGHT_COUNT vs AVG_INTERVAL")

fig, ax = plt.subplots()
ax.scatter(
    df["FLIGHT_COUNT"],
    df["AVG_INTERVAL"],
    c=df["Cluster"]
)
ax.set_xlabel("FLIGHT_COUNT")
ax.set_ylabel("AVG_INTERVAL")

st.pyplot(fig)

# ================= CLUSTER PROFILE =================
st.subheader("Cluster Profile (Mean Values)")
cluster_profile = df.groupby("Cluster")[df_numeric.columns].mean()
st.dataframe(cluster_profile)

# ================= EVALUATION =================
st.subheader("Silhouette Score")
score = silhouette_score(df_scaled, df["Cluster"])
st.write("Silhouette Score:", round(score, 4))