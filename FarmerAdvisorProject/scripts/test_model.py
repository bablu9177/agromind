import joblib
import pandas as pd
import numpy as np
import os

# Define paths
MODEL_PATH = r"C:\Users\kanth\OneDrive\Desktop\FarmerAdvisorProject\models\farmer_advisor_model.pkl"
ENCODER_PATH = r"C:\Users\kanth\OneDrive\Desktop\FarmerAdvisorProject\models\label_encoders.pkl"
TARGET_ENCODER_PATH = r"C:\Users\kanth\OneDrive\Desktop\FarmerAdvisorProject\models\y_encoder.pkl"
DATASET_PATH = r"C:\Users\kanth\OneDrive\Desktop\FarmerAdvisorProject\data\dataset.csv"

# Check if the model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Please train the model first.")

# Load trained model
model = joblib.load(MODEL_PATH)

# Load label encoders and target encoder
if os.path.exists(ENCODER_PATH):
    label_encoders = joblib.load(ENCODER_PATH)
else:
    raise FileNotFoundError(f"Label encoders file missing: {ENCODER_PATH}")

if os.path.exists(TARGET_ENCODER_PATH):
    y_encoder = joblib.load(TARGET_ENCODER_PATH)
else:
    raise FileNotFoundError(f"Target encoder file missing: {TARGET_ENCODER_PATH}")

# Load dataset for testing
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)

# Select a test sample
test_sample = df.iloc[0].copy()

# Encode categorical features
categorical_columns = ["Soil_Type", "Crop_Type", "Fertilizer_Used"]
for col in categorical_columns:
    if col in test_sample:
        test_sample[col] = label_encoders[col].transform([test_sample[col]])[0]

# Prepare features for prediction
test_features = np.array(test_sample.drop("Yield_Category")).reshape(1, -1)

# Predict yield category
y_pred = model.predict(test_features)
predicted_label = y_encoder.inverse_transform(y_pred)[0]

print(f"Predicted Yield Category: {predicted_label}")
