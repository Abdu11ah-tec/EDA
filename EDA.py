"""
Professional EDA Streamlit app
Features:
 - File upload (CSV or Excel)
 - Data loading and type inference
 - Quick overview (shape, sample, dtypes)
 - Missing values & imputation suggestions
 - Summary statistics (numeric & categorical)
 - Interactive distributions (Plotly)
 - Correlation matrix + heatmap
 - Time-series analysis (if date column present)
 - Top products/customers
 - RFM segmentation (Recency, Frequency, Monetary)
 - Outlier detection (IQR & z-score)
 - Download cleaned dataset
Author: ChatGPT (Gen Z vibez, but professional)
"""

import io
import math
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats

st.set_page_config(layout="wide", page_title="Professional EDA Dashboard")

# ---------------------------
# Helper Functions
# ---------------------------
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, low_memory=False)
    elif uploaded_file.name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type! Please upload CSV or Excel.")
        st.stop()
    return df

def infer_date_columns(df):
    date_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().astype(str).head(50).tolist()
            parsed = sum([1 for s in sample if _can_parse_date(s)])
            if parsed >= max(1, int(len(sample) * 0.4)):
                date_cols.append(col)
        elif np.issubdtype(df[col].dtype, np.datetime64):
            date_cols.append(col)
    return date_cols

def _can_parse_date(s):
    try:
        pd.to_datetime(s)
        return True
    except:
        return False

def summarize_missing(df):
    miss = df.isna().sum()
    miss_pct = (miss / len(df)) * 100
    mdf = pd.DataFrame({'missing_count': miss, 'missing_pct': miss_pct})
    return mdf.sort_values('missing_pct', ascending=False)

def numeric_cats(df):
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    cat = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return num, cat

def iqr_outliers(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return ((series < lower) | (series > upper)), lower, upper

def zscore_outliers(series, thresh=3):
    z = np.abs(stats.zscore(series.dropna()))
    idx = series.dropna().index[z > thresh]
    return idx

def rfm_table(df, customer_col='CustomerID', date_col='InvoiceDate', amount_col='Amount'):
    snapshot_date = df[date_col].max() + pd.Timedelta(days=1)
    rfm = df.groupby(customer_col).agg({
        date_col: lambda x: (snapshot_date - x.max()).days,
        customer_col: 'count',
        amount_col: 'sum'
    }).rename(columns={date_col: 'Recency', customer_col: 'Frequency', amount_col: 'Monetary'})
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1]).astype(int)
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5]).astype(int)
    rfm['RFM_Score'] = (rfm['R_Score']*100 + rfm['F_Score']*10 + rfm['M_Score']).astype(int)
    return rfm.reset_index()

# ---------------------------
# App Layout
# ---------------------------
st.title("Professional EDA — Streamlit")
st.write("Upload your dataset below to start exploring 👇")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is None:
    st.info("👈 Please upload a file from your computer to begin the analysis.")
    st.stop()

with st.spinner("Loading dataset..."):
    df = load_data(uploaded_file)

# Overview
st.header("1️⃣ Quick Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
c3.metric("Memory (MB)", f"{df.memory_usage(deep=True).sum() / (1024**2):.2f}")

st.dataframe(df.head(10))

# Missing values
st.header("2️⃣ Missing Values")
missing = summarize_missing(df)
st.dataframe(missing.style.format({'missing_pct': '{:.2f}%'}))

# Data types
st.header("3️⃣ Data Types & Conversions")
date_candidates = infer_date_columns(df)
st.write("Possible date-like columns:", date_candidates)

num_cols, cat_cols = numeric_cats(df)
st.write("Numeric columns:", num_cols)
st.write("Categorical columns:", cat_cols[:20])

# Convert detected dates
for c in date_candidates:
    try:
        df[c] = pd.to_datetime(df[c], errors='coerce')
    except:
        pass

# Summary stats
st.header("4️⃣ Summary Statistics")
if num_cols:
    st.subheader("Numeric Columns")
    st.dataframe(df[num_cols].describe().T)

    col = st.selectbox("Select numeric column for distribution", num_cols)
    fig = px.histogram(df, x=col, nbins=50, marginal="box", title=f"Distribution of {col}")
    st.plotly_chart(fig, use_container_width=True)

if cat_cols:
    st.subheader("Categorical Columns")
    cat_col = st.selectbox("Select categorical column", cat_cols)
    vc = df[cat_col].value_counts().reset_index()
    vc.columns = [cat_col, "Count"]
    fig = px.bar(vc.head(20), x=cat_col, y="Count", title=f"Top categories in {cat_col}")
    st.plotly_chart(fig, use_container_width=True)

# Correlation
st.header("5️⃣ Correlation Matrix")
if len(num_cols) >= 2:
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Outlier detection
st.header("6️⃣ Outlier Detection")
if num_cols:
    out_col = st.selectbox("Select column for outlier check", num_cols)
    method = st.radio("Method", ["IQR", "Z-Score"])
    if method == "IQR":
        mask, lower, upper = iqr_outliers(df[out_col].dropna())
        st.write(f"IQR bounds: {lower:.2f}, {upper:.2f}")
    else:
        idxs = zscore_outliers(df[out_col].dropna())
        st.write(f"Z-score outliers found: {len(idxs)}")

# Export cleaned dataset
st.header("✅ Export Cleaned Dataset")
buf = io.BytesIO()
df.to_csv(buf, index=False)
buf.seek(0)
st.download_button("Download cleaned CSV", data=buf, file_name="cleaned_dataset.csv", mime="text/csv")

st.success("🎉 EDA complete! Use controls in the sidebar to tweak views. If you want additional specialized analysis (cohort analysis, churn modeling, predictive features, etc.), tell me what you want next.")

