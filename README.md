# Water Quality ML Project

This repository contains code and structure for analysing water quality data in the context of climate change using machine learning.

## Structure
src/
  preprocess.py       # Preprocessing, reshaping, and feature engineering
data/
  raw/                # Local raw CSV (too large to upload)
  processed/          # Cleaned, imputed data outputs
reports/
  figures/            # Visualisations and graphs
## Team Roles
- **Kosi:** Data preprocessing, parameter selection, imputation, data quality analysis, assigneed WQI scores, NN modelling.
- **Noor:** 
- **Lucy:** MLR model building, outlier removal, PCA transformation, RF + KNN ML models
- **John:**

## Instructions
1. Place `water-quality-1.csv` in the `data/raw/` folder (not included in repo).
2. Run `preprocess_updated.py` to prepare the data.
3. Output will be saved to `data/processed/`.
4. Each member should maintain a short mini-report describing:
     - The task they completed (e.g preprocessing, visualisation, modelling).
     - The methods or libararies used.
     - Any key findings, observations or issues.

## Notes
Raw data is excluded for size and privacy reasons. Placeholders remain in the repo to indicate file structure.
Individual mini reports don't need to be fully formal or polished.
Use them for housekeeping, we can combine to make the final report for the upload.
Please upload under _reports/individual_.




