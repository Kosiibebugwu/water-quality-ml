import pandas as pd 
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# Load datase
df = pd.read_csv(r"C:\Users\kosii\OneDrive\Documents\water-quality-1.csv")

# Convert timestamps and derive temporal features
df['Collect DateTime'] = pd.to_datetime(
    df['Collect DateTime'],
    format='%m/%d/%Y %I:%M:%S %p',
    errors='coerce'
).dt.date

df['Year'] = pd.to_datetime(df['Collect DateTime']).dt.year
df['Month'] = pd.to_datetime(df['Collect DateTime']).dt.month

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

df['Season'] = df['Month'].apply(get_season)

# Keep only relevant WQI parameters
relevant_params = [
    "pH  Field", "Temperature", "Dissolved Oxygen  Field", "Conductivity  Field",
    "Total Nitrogen", "Total Phosphorus", "Nitrate Nitrogen", "Orthophosphate Phosphorus"
]

df_filtered = df[df["Parameter"].isin(relevant_params)].copy()

# Pivot to wide format
df_wide = df_filtered.pivot_table(
    index=['Sample ID', 'Collect DateTime', 'Year', 'Month', 'Season'],
    columns='Parameter',
    values='Value',
    aggfunc='mean'
).reset_index()

df_wide.columns.name = None
df_wide.columns = [str(c).strip().replace("  ", " ") for c in df_wide.columns]

# Rename and subset
cols_rename = {
    "pH Field": "pH",
    "Dissolved Oxygen Field": "DO",
    "Conductivity Field": "Conductivity",
    "Total Nitrogen": "Total_N",
    "Total Phosphorus": "Total_P",
    "Nitrate Nitrogen": "Nitrate",
    "Orthophosphate Phosphorus": "Orthophosphate"
}

df_small = df_wide.rename(columns=cols_rename).copy()

df_small = df_small[[
    'pH', 'DO', 'Conductivity', 'Temperature', 
    'Total_N', 'Total_P', 'Nitrate', 'Orthophosphate', 
    'Year', 'Month', 'Season'
]]

# Missingness Summary
missing_info = pd.DataFrame({
    "Missing Count": df_small.isna().sum(),
    "Missing %": (df_small.isna().sum() / len(df_small) * 100).round(2)
}).sort_values(by="Missing %", ascending=False)

print("\nMissingness Summary:\n", missing_info, "\n")

# Tiered Imputation
# Tier 1: KNN for moderately missing features
tier1_cols = ['Temperature', 'pH', 'DO', 'Conductivity']

imputer = KNNImputer(n_neighbors=5)
df_small[tier1_cols] = imputer.fit_transform(df_small[tier1_cols])

# Tier 2: Median/Seasonal for heavily missing values
tier2_cols = ['Total_N', 'Total_P', 'Orthophosphate']

for col in tier2_cols:
    df_small[col] = df_small.groupby('Season')[col].transform(lambda x: x.fillna(x.median()))
    df_small[col] = df_small[col].fillna(df_small[col].median())

# Tier 3: Minimal imputation for Nitrate (retain for low-weight WQI use)
df_small['Nitrate'] = df_small['Nitrate'].fillna(df_small['Nitrate'].median())

# Standardization (after imputation, before outlier removal)
scaler = StandardScaler()
all_numeric = ['Temperature', 'pH', 'DO', 'Conductivity', 'Total_N', 'Total_P', 'Orthophosphate', 'Nitrate']
df_small[all_numeric] = scaler.fit_transform(df_small[all_numeric])

# Check
print("\nPost-Imputation & Scaling Summary:")
print(df_small.describe().T)

# Save Cleaned Dataset
output_path = r"C:\Users\kosii\OneDrive\Documents\water-quality-cleaned.csv"
df_small.to_csv(output_path, index=False)
print(f"\n Clean dataset saved as: {output_path}")