
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "food_demand_sample.csv"
MODEL_DIR = BASE_DIR
MODEL_PATH = MODEL_DIR / "food_demand_model.pkl"


# Load the existing dataset
df = pd.read_csv(DATA_PATH)

# Features used to predict meals consumed (demand)
features = [
    "day_of_week",
    "temperature_c",
    "rainfall_mm",
    "special_event"
]
target = "meals_consumed"

# Keep the required columns and remove rows without a target value
df = df[features + [target]].copy()
df[target] = pd.to_numeric(df[target], errors="coerce")
df = df.dropna(subset=[target])

X = df[features]
y = df[target]

# Prepare numerical and categorical features
categorical_features = ["day_of_week", "special_event"]
numerical_features = ["temperature_c", "rainfall_mm"]

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ]), numerical_features),

    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

# Build the prediction model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# Split data for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

# Save the trained model
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("\nModel training completed!")
print(f"Rows used: {len(df)}")
print(f"MAE: {mae:.2f} meals")
print(f"RMSE: {rmse:.2f} meals")
print(f"R² score: {r2:.3f}")
print(f"Saved model to: {MODEL_PATH}")
