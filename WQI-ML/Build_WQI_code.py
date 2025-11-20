#make WQI formula using a modified NSF-WQI
#use parameters from cleaned water quality cvs
#Convert each parameter → 0–100 “sub-index”

import pandas as pd
import numpy as np

# 1. Load your cleaned dataset
df = pd.read_csv(r"C:\Users\kosii\Downloads\water-quality-cleaned.csv")
df.columns = df.columns.str.strip()

# 2. Define scoring functions
def score_do(val):
    # if val is standardized, this is a made-up linear scaling: higher = better
    return np.clip(20*val + 50, 0, 100)

def score_ph(val):
    ideal = 7.5
    #if pH is standardized, this is not chemically correct, but it will run
    return np.clip(100 - abs(val - ideal)*20, 0, 100)

def score_temp(t):
    return np.clip(100 - (t-20)*4, 0, 100)

def inverse_score(val, scale=50):
    return np.clip(100 - max(0, val*scale), 0, 100)

# 3. create the score columns first
df["DO_score"]       = df["DO"].apply(score_do)
df["pH_score"]       = df["pH"].apply(score_ph)
df["Temp_score"]     = df["Temperature"].apply(score_temp)
df["Nitrate_score"]  = df["Nitrate"].apply(lambda x: inverse_score(x, scale=120))
df["TotalN_score"]   = df["Total_N"].apply(lambda x: inverse_score(x, scale=60))
df["TotalP_score"]   = df["Total_P"].apply(lambda x: inverse_score(x, scale=150))
df["Cond_score"]     = df["Conductivity"].apply(lambda x: inverse_score(x, scale=40))
df["OrthoP_score"]   = df["Orthophosphate"].apply(lambda x: inverse_score(x, scale=120))

# 4. Define weights
weights = {
    "DO_score": 0.2,
    "pH_score": 0.2,
    "Temp_score": 0.15,
    "Nitrate_score": 0.1,
    "TotalN_score": 0.1,
    "TotalP_score": 0.1,
    "Cond_score": 0.1,
    "OrthoP_score": 0.05,
}

# 5. Compute WQI from those score columns
df["WQI"] = (
    df["DO_score"]      * weights["DO_score"] +
    df["pH_score"]      * weights["pH_score"] +
    df["Temp_score"]    * weights["Temp_score"] +
    df["Nitrate_score"] * weights["Nitrate_score"] +
    df["TotalN_score"]  * weights["TotalN_score"] +
    df["TotalP_score"]  * weights["TotalP_score"] +
    df["Cond_score"]    * weights["Cond_score"] +
    df["OrthoP_score"]  * weights["OrthoP_score"]
)

print(df[["WQI"]].head())

df.to_csv(r"C:\Users\kosii\OneDrive\Documents\cleaned_with_WQI.csv", index=False)
print ("WQI saved in downloads") #do this to see if works or check your donwloads in File Explorer