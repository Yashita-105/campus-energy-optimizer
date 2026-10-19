import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/processed/campus_energy_real.csv"
OUTPUT_PATH = "data/processed/campus_energy_pred.csv"


df = pd.read_csv(DATA_PATH)

features = [
    "temperature",
    "humidity",
    "occupancy",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "room_type",
]
target = "energy"

X = df[features]
y = df[target]

numeric_features = [
    "temperature", "humidity", "occupancy", "hour",
    "day_of_week", "month", "is_weekend",
]
categorical_features = ["room_type"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), categorical_features),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", Ridge(alpha=1.0)),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Model Evaluation")
print("MAE:", round(mae, 3))
print("R2 :", round(r2, 3))

# Predict all observations for the decision agent.
df["pred_energy"] = model.predict(X)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Predictions saved to {OUTPUT_PATH}")
