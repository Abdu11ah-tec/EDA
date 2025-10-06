# app.py
"""
Professional EDA Streamlit app for: /mnt/data/ecommerce_dataset.csv
Features:
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

import os
import io
import math
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

import streamlit as st
from PIL import Image

# plotting
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# utils
from sklearn.preprocessing import StandardScaler
from scipy import stats

st.set_page_config(layout="wide", page_title="Professional EDA Dashboard")

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    return df

def infer_date_columns(df):
    date_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().astype(str).head(50).tolist()
            # quick heuristic: try parsing a sample
            parsed = 0
            for s in sample:
                try:
                    pd.to_datetime(s)
                    parsed += 1
                except:
                    pass
            if parsed >= max(1, int(len(sample) * 0.4)):
                date_cols.append(col)
        elif np.issubdtype(df[col].dtype, np.datetime64):
            date_cols.append(col)
    return date_cols

def summarize_missing(df):
    miss = df.isna().sum()
    miss_pct = (miss / len(df)) * 100
    mdf = pd.DataFrame({'missing_count': miss, 'missing_pct': miss_pct})
    mdf = mdf.sort_values('missing_pct', ascending=False)
    return mdf

def numeric_cats(df):
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    cat = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return num, cat

def iqr_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return ((series < lower) | (series > upper)), lower, upper

def zscore_outliers(series, thresh=3):
    z = np.abs(stats.zscore(series.dropna()))
    idx = series.dropna().index[z > thresh]
    return idx

def rfm_table(df, customer_col='CustomerID', date_col='InvoiceDate', amount_col='Amount'):
    # requires datetime date_col and numeric amount_col
    snapshot_date = df[date_col].max() + pd.Timedelta(days=1)
    rfm = df.groupby(customer_col).agg({
        date_col: lambda x: (snapshot_date - x.max()).days,
        customer_col: 'count',
        amount_col: 'sum'
    }).rename(columns={date_col: 'Recency', customer_col: 'Frequency', amount_col: 'Monetary'})
    # scores (1-5)
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1]).astype(int)
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5]).astype(int)
    rfm['RFM_Score'] = (rfm['R_Score']*100 + rfm['F_Score']*10 + rfm['M_Score']).astype(int)
    return rfm.reset_index()

# ---------------------------
# App layout
# ---------------------------
st.title("Professional EDA — Streamlit")
st.write("Performing a comprehensive Exploratory Data Analysis (EDA). File: `/mnt/data/ecommerce_dataset.csv`")

# show small logo / image for vibe (optional)
# try to load a small icon if available
try:
    img = Image.open(io.BytesIO())
except:
    pass

# Sidebar - controls
st.sidebar.header("Controls")
data_path = st.sidebar.text_input("CSV path", value="/mnt/data/ecommerce_dataset.csv")
show_sample = st.sidebar.checkbox("Show data sample", value=True)
nrows = st.sidebar.number_input("Rows to show (sample)", min_value=5, max_value=500, value=10)

# Load
if not os.path.exists(data_path):
    st.error(f"File not found at `{data_path}`. Make sure the path is correct.")
    st.stop()

with st.spinner("Loading dataset..."):
    df = load_data(data_path)

# quick overview
st.header("1) Quick Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
mem = df.memory_usage(deep=True).sum() / (1024**2)
c3.metric("Memory (MB)", f"{mem:.2f}")

if show_sample:
    st.subheader("Sample")
    st.dataframe(df.head(int(nrows)))

with st.expander("Column types & non-null counts"):
    dtypes = pd.DataFrame({
        'dtype': df.dtypes.astype(str),
        'non_null_count': df.notna().sum(),
        'unique_count': df.nunique(dropna=False)
    }).sort_values('dtype')
    st.dataframe(dtypes)

# missingness
st.header("2) Missing Values")
missing = summarize_missing(df)
st.dataframe(missing.style.format({'missing_pct': '{:.2f}%'}).head(200))

st.markdown("#### Missingness suggestions")
top_missing = missing[missing['missing_count'] > 0]
for col, row in top_missing.head(10).iterrows():
    pct = row['missing_pct']
    suggestion = "No action required" if pct < 1 else (
        "Consider imputation (mean/median) for numeric or mode for categorical" if pct < 30 else
        "Consider dropping column or collecting more data; high missingness"
    )
    st.write(f"**{col}** — {row['missing_count']} missing ({row['missing_pct']:.2f}%) → {suggestion}")

# data types
st.header("3) Data Types & Suggested Conversions")
date_candidates = infer_date_columns(df)
st.write("Detected date-like columns:", date_candidates)
if date_candidates:
    st.markdown("You can convert these columns to datetime. Example:")
    sample_date_col = date_candidates[0]
    st.code(f"df['{sample_date_col}'] = pd.to_datetime(df['{sample_date_col}'], errors='coerce')")

# show numeric and categorical
num_cols, cat_cols = numeric_cats(df)
st.write("Numeric columns detected:", num_cols)
st.write("Categorical columns detected:", cat_cols[:20])

# convert common columns if present
st.header("4) Automatic Conversions (applied in-memory view only)")
to_convert = []
for c in date_candidates:
    try:
        df[c] = pd.to_datetime(df[c], errors='coerce')
        to_convert.append(c)
    except Exception:
        pass
if to_convert:
    st.success(f"Converted to datetime: {to_convert}")

# create a standardized amount column if possible (often Total, Amount, Price, UnitPrice)
possible_amounts = [c for c in df.columns if any(keyword in c.lower() for keyword in ['amount','price','total','revenue','sales','ordervalue'])]
st.write("Possible monetary columns:", possible_amounts)

# if no explicit amount and there are UnitPrice & Quantity, create Amount
if 'Amount' not in df.columns:
    unit_cols = [c for c in df.columns if 'unit' in c.lower() or 'price' in c.lower()]
    qty_cols = [c for c in df.columns if 'qty' in c.lower() or 'quantity' in c.lower() or 'units' in c.lower()]
    # heuristic:
    if unit_cols and qty_cols:
        u = unit_cols[0]
        q = qty_cols[0]
        try:
            df['Amount'] = pd.to_numeric(df[u], errors='coerce') * pd.to_numeric(df[q], errors='coerce')
            st.info(f"Created 'Amount' as {u} * {q}")
        except Exception:
            pass

# ---------------------------
# Summary stats & distributions
# ---------------------------
st.header("5) Summary Statistics & Distributions")

if num_cols:
    st.subheader("Numeric summary")
    st.dataframe(df[num_cols].describe().T)

    # interactive histograms
    st.subheader("Interactive Numeric Distributions")
    col = st.selectbox("Choose numeric column for distribution", options=num_cols)
    if col:
        fig = px.histogram(df, x=col, nbins=50, marginal="box", title=f"Distribution of {col}")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

# categorical summaries
if cat_cols:
    st.subheader("Categorical summary (top values)")
    cat_choice = st.selectbox("Choose categorical column to inspect", options=cat_cols, index=0)
    if cat_choice:
        vc = df[cat_choice].value_counts(dropna=False).reset_index()
        vc.columns = [cat_choice, 'count']
        st.dataframe(vc.head(100))
        if vc.shape[0] <= 50:
            fig = px.bar(vc.head(50), x=cat_choice, y='count', title=f"Value counts - {cat_choice}")
            fig.update_layout(xaxis={'categoryorder':'total descending'}, height=400)
            st.plotly_chart(fig, use_container_width=True)

# correlation
st.header("6) Correlation Matrix")
numeric_for_corr = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_for_corr) >= 2:
    corr_matrix = df[numeric_for_corr].corr()
    st.write("Top correlations (absolute) — inspect for multicollinearity:")
    # show top correlated pairs
    corr_pairs = (
        corr_matrix.abs().stack()
        .reset_index()
        .query("level_0 != level_1")
        .rename(columns={0:'abs_corr'})
        .sort_values('abs_corr', ascending=False)
    )
    st.dataframe(corr_pairs.head(20))

    # heatmap (matplotlib/seaborn)
    fig, ax = plt.subplots(figsize=(min(14, len(numeric_for_corr)), min(10, len(numeric_for_corr))))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='RdYlBu_r', center=0, ax=ax)
    ax.set_title("Correlation heatmap")
    st.pyplot(fig)
else:
    st.info("Not enough numeric columns for correlation matrix.")

# ---------------------------
# Time-series & cohort checks
# ---------------------------
st.header("7) Time-series / Temporal Analysis")
date_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
if date_cols:
    st.write("Datetime columns available:", date_cols)
    dt_col = st.selectbox("Choose datetime column for time-series plots", options=date_cols)
    if dt_col:
        # ensure datetime index
        tmp = df[[dt_col]].copy()
        tmp[dt_col] = pd.to_datetime(tmp[dt_col], errors='coerce')
        tmp['__count__'] = 1
        freq = st.selectbox("Aggregation frequency", options=['D', 'W', 'M', 'Q', 'Y'], index=2)
        ts = tmp.set_index(dt_col).resample(freq)['__count__'].sum().rename('count').reset_index()
        fig = px.line(ts, x=dt_col, y='count', title=f"Records over time ({freq})")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        # if Amount present, show revenue over time
        if 'Amount' in df.columns:
            tmp2 = df[[dt_col,'Amount']].copy()
            tmp2[dt_col] = pd.to_datetime(tmp2[dt_col], errors='coerce')
            rev = tmp2.set_index(dt_col).resample(freq)['Amount'].sum().reset_index()
            fig2 = px.line(rev, x=dt_col, y='Amount', title=f"Revenue over time ({freq})")
            st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No datetime columns found. Consider converting columns that look like dates.")

# ---------------------------
# Top items / customers
# ---------------------------
st.header("8) Top products & customers")
customer_cols = [c for c in df.columns if 'customer' in c.lower()]
product_cols = [c for c in df.columns if any(k in c.lower() for k in ['product','item','sku','description','stockcode'])]
st.write("Detected customer columns:", customer_cols)
st.write("Detected product columns:", product_cols)

if customer_cols:
    cust = customer_cols[0]
    st.subheader(f"Top customers by revenue (column: {cust})")
    if 'Amount' in df.columns:
        top_cust = df.groupby(cust)['Amount'].sum().sort_values(ascending=False).reset_index().head(20)
        fig = px.bar(top_cust, x=cust, y='Amount', title="Top customers by Amount")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top_cust)
    else:
        st.write("No 'Amount' column to aggregate. You can create Amount from UnitPrice * Quantity if present.")

if product_cols:
    prod = product_cols[0]
    st.subheader(f"Top products by frequency (column: {prod})")
    top_prod = df[prod].value_counts().reset_index().rename(columns={'index':prod, prod:'count'}).head(20)
    fig = px.bar(top_prod, x=prod, y='count', title="Top products by count")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(top_prod)

# ---------------------------
# RFM segmentation
# ---------------------------
st.header("9) RFM Segmentation (If applicable)")
# heuristics to pick columns
customer_col = customer_cols[0] if customer_cols else None
date_col = date_cols[0] if date_cols else None
amount_col = 'Amount' if 'Amount' in df.columns else None

if customer_col and date_col and amount_col:
    st.write(f"Using Customer: `{customer_col}`, Date: `{date_col}`, Amount: `{amount_col}`")
    # ensure types
    tmp = df[[customer_col, date_col, amount_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors='coerce')
    tmp = tmp.dropna(subset=[customer_col, date_col])
    rfm = rfm_table(tmp, customer_col=customer_col, date_col=date_col, amount_col=amount_col)
    st.dataframe(rfm.head(50))
    fig = px.histogram(rfm, x='RFM_Score', nbins=20, title="RFM Score Distribution")
    st.plotly_chart(fig, use_container_width=True)
    # show top segments
    top_rfm = rfm.sort_values('RFM_Score', ascending=False).head(20)
    st.subheader("Top RFM customers")
    st.dataframe(top_rfm)
else:
    st.info("RFM needs customer id, datetime, and amount columns. Detected: "
            f"customer_col={customer_col}, date_col={date_col}, amount_col={amount_col}")

# ---------------------------
# Outlier detection
# ---------------------------
st.header("10) Outlier Detection")
if num_cols:
    out_col = st.selectbox("Choose numeric column for outlier detection", options=num_cols, index=0)
    method = st.selectbox("Method", options=['IQR', 'z-score'], index=0)
    if method == 'IQR':
        mask, lower, upper = iqr_outliers(df[out_col].dropna())
        outliers = df[out_col].loc[mask.index[mask]].dropna()
        st.write(f"IQR bounds for `{out_col}`: lower={lower:.3g}, upper={upper:.3g}")
        st.write(f"Outlier count (IQR): {outliers.shape[0]}")
        if outliers.shape[0] > 0:
            st.dataframe(df.loc[outliers.index].head(200))
    else:
        idxs = zscore_outliers(df[out_col].dropna())
        st.write(f"Outlier count (z-score>3): {len(idxs)}")
        if len(idxs) > 0:
            st.dataframe(df.loc[idxs].head(200))
else:
    st.info("No numeric columns for outlier detection.")

# ---------------------------
# Data quality checks / suggestions
# ---------------------------
st.header("11) Data quality & next steps (automated suggestions)")
suggestions = []
# dup check
dup_count = df.duplicated().sum()
suggestions.append(f"Duplicate rows: {dup_count} — consider dropping duplicates if they are exact replicas.")
# missing columns suggestion
high_missing = missing[missing['missing_pct'] > 50]
if not high_missing.empty:
    suggestions.append(f"Columns with >50% missing: {', '.join(high_missing.index.astype(str).tolist()[:10])}")
# dtype suggestion
if any(col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() > 1000):
    suggestions.append("Large-cardinality object columns detected (likely ids). Consider converting to categorical/object expected types.")
# amount suggestion
if 'Amount' not in df.columns and possible_amounts:
    suggestions.append("No explicit 'Amount' column detected — consider creating Amount = UnitPrice * Quantity if applicable.")
# timestamp suggestion
if not date_cols:
    suggestions.append("No datetime columns detected. Time-based analyses will be limited unless you convert date-like columns to datetime.")
# show suggestions
for s in suggestions:
    st.write("- " + s)

# ---------------------------
# Export cleaned/converted dataset
# ---------------------------
st.header("12) Export cleaned dataset")
export_cols = st.multiselect("Select columns to include in exported CSV (leave blank for all)", options=df.columns.tolist())
if len(export_cols) == 0:
    export_df = df.copy()
else:
    export_df = df[export_cols].copy()

buf = io.BytesIO()
export_df.to_csv(buf, index=False)
buf.seek(0)
st.download_button("Download CSV of cleaned/explored data", data=buf, file_name="cleaned_data.csv", mime="text/csv")

st.success("EDA complete. Use controls in the sidebar to tweak views. If you want additional specialized analysis (cohort analysis, churn modeling, predictive features, etc.), tell me what you want next.")
