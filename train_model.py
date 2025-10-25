# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load Hyderabad data
df = pd.read_csv("Hyderabad_House_Data.csv")
df = df.dropna()

# Feature engineering
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]) if isinstance(x, str) else None)
df = df[df['total_sqft'].apply(lambda x: str(x).replace('.', '', 1).isdigit())]
df['total_sqft'] = df['total_sqft'].astype(float)
df = df[df['bath'] < df['bhk'] + 2]
df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']

# Encode categorical location
df['location'] = df['location'].apply(lambda x: x.strip())
location_stats = df['location'].value_counts()
# Keep all locations, don't group them as 'other'
dummies = pd.get_dummies(df['location'], prefix='location')
df = pd.concat([df.drop('location', axis=1), dummies], axis=1)

# Remove the 'size' column as it's not needed for training
df = df.drop('size', axis=1)

# Train-test split
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Hyderabad Model Trained Successfully!")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")

# Save model and metadata
joblib.dump(model, "house_price_model.pkl")
joblib.dump(X.columns, "model_columns.pkl")
metrics = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
joblib.dump(metrics, "model_metrics.pkl")

print("\nModel, Columns, and Metrics Saved Successfully for Hyderabad!")
