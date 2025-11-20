import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models

import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\kosii\OneDrive\Documents\cleaned_with_WQI.csv")
df.columns = df.columns.str.strip()

print("Columns:")
print(df.columns)

# Encode 'Season' because neural networks can only process numeric inputs.
# Without encoding, the NN would ignore or fail on the non-numeric season labels.
df = pd.get_dummies(df, columns=["Season"], drop_first=True)

print("New columns after encoding:")
print(df.columns)

#define th features as (X) and the target as (Y)
X = df.drop(["WQI"], axis=1)
y = df["WQI"]

print("X shape:", X.shape)
print("y shape:", y.shape)

#train/test split 80:20
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

#standardise features (NN requirement)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

#build NN
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # output = predicted WQI
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

#train NN
history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1
)

#plots
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('MAE Over Epochs')
plt.legend()

plt.show()

#evaluate NN w/test data
test_loss, test_mae = model.evaluate(X_test_scaled, y_test)
print("Test MSE:", test_loss)
print("Test MAE:", test_mae)

#save the model
model.save("nn_wqi_model.h5")
print("Model saved as nn_wqi_model.h5")
