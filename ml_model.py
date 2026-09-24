import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
# Load Data
df = pd.read_csv("campus_energy_big.csv")
# Encode Categorical Features
encoder = OneHotEncoder(sparse_output=False)
cat_encoded = encoder.fit_transform(df[["room_type", "ac_level"]])

cat_df = pd.DataFrame(
    cat_encoded,
    columns=encoder.get_feature_names_out(["room_type", "ac_level"])
)

# Feature Matrix (same features as before)
X = pd.concat(
    [df[["temperature", "humidity", "occupancy", "hour"]], cat_df],
    axis=1
)

y = df["energy"]

# Train-Test Split (internal improvement)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling (improves stability)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ridge Regression (regularized)

model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# Evaluation (for review discussion)

y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print("MAE:", round(mae, 2))
print("R2 Score:", round(r2, 3))


# Predict FULL dataset (IMPORTANT)
X_full_scaled = scaler.transform(X)
df["pred_energy"] = model.predict(X_full_scaled)

# Save SAME file format as before
df.to_csv("campus_energy_big_pred.csv", index=False)
print("Upgraded ML model trained & predictions saved.")