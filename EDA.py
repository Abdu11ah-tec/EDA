# ===============================
# 1. Import Libraries
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import missingno as msno

# ===============================
# 2. Load Data
# ===============================
df = pd.read_csv("ecommerce_dataset.csv")

# ===============================
# 3. Data Cleaning
# ===============================

# Convert order_date to datetime
df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

# Drop duplicate rows
df = df.drop_duplicates()

# Handle missing values
# Numeric -> fill with median
for col in df.select_dtypes(include=np.number).columns:
    df[col].fillna(df[col].median(), inplace=True)

# Categorical -> fill with mode
for col in df.select_dtypes(include="object").columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Remove outliers using Z-score (only numeric cols)
numeric_cols = df.select_dtypes(include=np.number).columns
z_scores = np.abs(stats.zscore(df[numeric_cols]))
df = df[(z_scores < 3).all(axis=1)]  # keep rows where all z-scores < 3

print(f"Cleaned dataset shape: {df.shape}")

# ===============================
# 4. Exploratory Data Analysis
# ===============================

# ---- Missingness Visualization
msno.matrix(df)
plt.show()

# ---- Descriptive statistics
print("\nSummary Statistics:")
print(df.describe(include='all').T)

# ---- Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ---- Histograms
df[numeric_cols].hist(figsize=(12, 8), bins=20)
plt.suptitle("Histograms of Numeric Features")
plt.show()

# ---- Boxplots for numeric features
plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot - {col}")
plt.tight_layout()
plt.show()

# ---- Categorical distributions
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    plt.figure(figsize=(8,4))
    sns.countplot(y=col, data=df, order=df[col].value_counts().index)
    plt.title(f"Distribution of {col}")
    plt.show()

# ---- Time series: orders over time
plt.figure(figsize=(12, 6))
df.set_index('order_date').resample('D').size().plot()
plt.title("Orders Over Time")
plt.ylabel("Number of Orders")
plt.show()

# ---- Revenue over time
df['revenue'] = df['quantity'] * df['price'] * (1 - df['discount'])
plt.figure(figsize=(12, 6))
df.set_index('order_date')['revenue'].resample('M').sum().plot()
plt.title("Monthly Revenue Trend")
plt.ylabel("Revenue")
plt.show()

# ---- Regional sales breakdown
plt.figure(figsize=(8,4))
sns.barplot(x='region', y='revenue', data=df, estimator=sum, ci=None)
plt.title("Revenue by Region")
plt.show()

# ---- Payment method popularity
plt.figure(figsize=(8,4))
sns.countplot(x='payment_method', data=df, order=df['payment_method'].value_counts().index)
plt.title("Payment Method Distribution")
plt.show()
