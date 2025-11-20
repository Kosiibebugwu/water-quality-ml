# Water Quality Index ML Project

1. Use `cleaned_with_WQI.csv` as the dataset.
2. Target column: WQI
3. Features = all columns except WQI
4. One-hot encode Season:

   df = pd.get_dummies(df, columns=["Season"], drop_first=True)

5. Use the same split for all models:

   train_test_split(X, y, test_size=0.2, random_state=42)

6. Scale features (StandardScaler) for NN and SVM. XGBoost can be unscaled.

7. My NN implementation is in nn_workflow.py.
