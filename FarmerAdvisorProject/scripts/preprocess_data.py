import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/dataset.csv")

# Handling missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Encoding categorical variables
label_encoders = {}
categorical_cols = ["soil_type", "NPK_ratio", "water_availability", "crop_preference"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encoding target variable (Crop Recommendation)
crop_encoder = LabelEncoder()
df["recommended_crop"] = crop_encoder.fit_transform(df["recommended_crop"])

# Splitting the dataset
X = df.drop(columns=["recommended_crop"])
y = df["recommended_crop"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save processed data and encoders
df.to_csv("data/processed_dataset.csv", index=False)
joblib.dump(label_encoders, "models/label_encoders.pkl")
joblib.dump(crop_encoder, "models/crop_encoder.pkl")

print("Preprocessing complete. Processed data and encoders saved.")
