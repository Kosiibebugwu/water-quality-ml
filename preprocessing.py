import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

df = pd.read_csv(r"C:\Users\kosii\OneDrive\Documents\water-quality-1.csv")

# Convert American-style timestamps to datetime (ignore hour/min)
df['Collect DateTime'] = pd.to_datetime(
    df['Collect DateTime'],
    format='%m/%d/%Y %I:%M:%S %p',
    errors='coerce'
).dt.date

# Derive temporal features
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


#Keep only relevant WQI parameters
relevant_params = [
    "pH  Field", "Temperature", "Dissolved Oxygen  Field", "Conductivity  Field",
    "Total Nitrogen", "Total Phosphorus", "Nitrate Nitrogen", "Orthophosphate Phosphorus"
]

df_filtered = df[df["Parameter"].isin(relevant_params)].copy()

print("Filtered dataset shape:", df_filtered.shape)
print("Unique parameters kept:", df_filtered["Parameter"].unique())


#Pivot to wide format
df_wide = df_filtered.pivot_table(
    index=['Sample ID', 'Collect DateTime', 'Year', 'Month', 'Season'],
    columns='Parameter',
    values='Value',
    aggfunc='mean'
).reset_index()

#Simplify column names
df_wide.columns.name = None
df_wide.columns = [str(c).strip().replace("  ", " ") for c in df_wide.columns]

print("Wide dataset shape:", df_wide.shape)
print("Columns:", df_wide.columns.tolist())


#Rename and subset for analysis
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

#Reorder columns neatly
df_small = df_small[[
    'pH', 'DO', 'Conductivity', 'Temperature', 'Total_N',
    'Total_P', 'Nitrate', 'Orthophosphate', 'Year', 'Month', 'Season'
]]

print(df_small.head())
print(df_small.shape)


# Missingness summary
missing_info = pd.DataFrame({
    "Missing Count": df_small.isna().sum(),
    "Missing %": (df_small.isna().sum() / len(df_small) * 100).round(2)
}).sort_values(by="Missing %", ascending=False)

print(missing_info)

#Outlier removal
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

numeric_cols = ['Temperature', 'pH', 'DO', 'Conductivity', 
                'Total_N', 'Total_P', 'Nitrate', 'Orthophosphate']

for col in numeric_cols:
    df_small = remove_outliers_iqr(df_small, col)

#KNN imputation (Temperature only)
imputer = KNNImputer(n_neighbors=5)
df_small['Temperature'] = imputer.fit_transform(df_small[['Temperature']])

#PRE-1990 TEMPERTURE SPIKE
year_counts = df_small['Year'].value_counts().sort_index()
print(year_counts)

#Visualise data availability
year_counts.plot(kind='bar', title='Sample Count by Year')

#Check descriptive stats for Temperature by year
yearly_stats = df_small.groupby('Year')['Temperature'].describe()
print(yearly_stats.loc[1980:1995])  # focus on period of interest

#Quick visual of the spike
df_small.groupby('Year')['Temperature'].median().plot(title='Yearly Median Temperature')
