# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 23:49:30 2026

@author: Barney
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# === Set seeds for reproducibility ===
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds(42)

# === Load surface data ===
df_surface = pd.read_excel("Compressive_data_LIG.xlsx", header=0)
x = df_surface.iloc[:, 0].values
y = df_surface.iloc[:, 1].values
z = df_surface.iloc[:, 3].values
X = np.column_stack((x, y))

# === Scale data ===
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_z = StandardScaler()
z_scaled = scaler_z.fit_transform(z.reshape(-1,1)).ravel()

# === Split data ===
X_train, X_test, z_train, z_test = train_test_split(
    X_scaled, z_scaled, test_size=0.2, random_state=42
)

# === Model parameters ===
best_lr = 0.00075
num_epochs = 150
batch_size = 16

# === Build and train model ===
set_seeds(42)
model = Sequential([
    Dense(32, activation='relu', input_shape=(2,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=best_lr), loss='mse')

history = model.fit(
    X_train, z_train,
    validation_data=(X_test, z_test),
    epochs=num_epochs,
    batch_size=batch_size,
    verbose=0
)

# === Save the trained model and scalers ===
model.save("compressive_model.h5", include_optimizer=False)
import joblib
joblib.dump(scaler_X, "scaler_X.save")
joblib.dump(scaler_z, "scaler_z.save")
print("Model and scalers saved successfully!")