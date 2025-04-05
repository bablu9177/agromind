import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier  # You can change this model

# Define file paths
DATASET_PATH = r"C:\Users\kanth\OneDrive\Desktop\FarmerAdvisorProject\data\dataset.csv"
MODEL_PATH = r"C:\Users\kanth\OneDrive\Desktop\FarmerAdvisorProject\models\farmer_advisor_model.pkl"
ENCODER_PATH = r"C:\Users\kanth\OneDrive\Desktop\FarmerAdvisorProject\models\label_encoders.pkl"
TARGET_ENCODER_PATH = r"C:\Users\kanth\OneDrive\Desktop\FarmerAdvisorProject\models\y_encoder.pkl"

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Define categorical and numerical columns
categorical_columns = ["Soil_Type", "Crop_Type", "Fertilizer_Used"]
target_column = "Yield_Category"

# Encode categorical variables
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode target variable
y_encoder = LabelEncoder()
df[target_column] = y_encoder.fit_transform(df[target_column])

# Split features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Normalize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model and encoders
joblib.dump(model, MODEL_PATH)
joblib.dump(label_encoders, ENCODER_PATH)
joblib.dump(y_encoder, TARGET_ENCODER_PATH)

print("✅ Model trained and saved successfully!")
