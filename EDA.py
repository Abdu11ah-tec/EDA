import pandas as pd
import plotly.express as px

# -------------------
# Load Dataset
# -------------------
df = pd.read_csv("ecommerce_dataset.csv")
df['order_date'] = pd.to_datetime(df['order_date'])
df['month'] = df['order_date'].dt.to_period('M').astype(str)

# -------------------
# Basic Info
# -------------------
print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nSummary Stats:\n", df.describe())

# -------------------
# Category Distribution
# -------------------
fig = px.histogram(df, x="category", title="Product Category Distribution")
fig.show()

# -------------------
# Price Distribution
# -------------------
fig = px.histogram(df, x="price", nbins=30, marginal="box",
                   title="Price Distribution with Boxplot")
fig.show()

# -------------------
# Price vs Quantity by Category
# -------------------
fig = px.scatter(df, x="price", y="quantity", color="category",
                 hover_data=["order_id", "customer_id"],
                 title="Price vs Quantity by Category")
