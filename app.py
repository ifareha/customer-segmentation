import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("🛒 Online Retail Customer Segmentation")
st.write("RFM Analysis + K-Means Clustering")

# ======================
# Upload Dataset
# ======================
uploaded_file = st.file_uploader("Upload Online Retail CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding="latin1")

    st.subheader("📄 Raw Dataset Preview")
    st.dataframe(df.head())

    # ======================
    # Data Cleaning
    # ======================
    df = df.dropna(subset=['CustomerID'])
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    # ======================
    # RFM Calculation
    # ======================
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'UnitPrice': 'sum'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    st.subheader("📊 RFM Table")
    st.dataframe(rfm.head())

    # ======================
    # Scaling
    # ======================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm)

    # ======================
    # Choose K
    # ======================
    st.subheader("⚙️ Select Number of Clusters")
    k = st.slider("Number of Clusters (K)", min_value=2, max_value=10, value=4)

    # ======================
    # Train KMeans
    # ======================
    kmeans = KMeans(n_clusters=k, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(X_scaled)

    # ======================
    # Cluster Summary
    # ======================
    st.subheader("📌 Cluster Summary")
    st.dataframe(
        rfm.groupby('Cluster').agg(['mean', 'count'])
    )

    # ======================
    # Visualizations
    # ======================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Recency vs Monetary")
        fig1, ax1 = plt.subplots()
        sns.scatterplot(
            x=rfm['Recency'],
            y=rfm['Monetary'],
            hue=rfm['Cluster'],
            palette="Set2",
            ax=ax1
        )
        st.pyplot(fig1)

    with col2:
        st.subheader("Frequency vs Monetary")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(
            x=rfm['Frequency'],
            y=rfm['Monetary'],
            hue=rfm['Cluster'],
            palette="Set2",
            ax=ax2
        )
        st.pyplot(fig2)


else:
    st.info("Please upload Online_Retail.csv to start")
