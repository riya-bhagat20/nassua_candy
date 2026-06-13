import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy import stats

# Load data
df = pd.read_csv("nassau_candy_orders.csv")

print("Shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nNull counts:\n", df.isnull().sum())
print("\nSample:\n", df.head())

# Drop rows with critical nulls
df.dropna(subset=["factory_id", "product_id", "region", "ship_mode",
                   "shipping_distance_miles", "lead_time_days", "margin"], inplace=True)
df.reset_index(drop=True, inplace=True)





import plotly.express as px
import plotly.graphics as gs
