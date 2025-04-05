import joblib
import pandas as pd

# Load the trained model and encoders
model = joblib.load("models/farmer_advisor_model.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")
crop_encoder = joblib.load("models/crop_encoder.pkl")

# Example new farmer input
new_farm = {
    "soil_type": "Loamy",
    "pH_level": 6.5,
    "NPK_ratio": "High",
    "water_availability": "High",
    "crop_preference": "Vegetables"
}

# Encode the input
encoded_input = []
for key, value in new_farm.items():
    if key in label_encoders:
        encoded_input.append(label_encoders[key].transform([value])[0])
    else:
        encoded_input.append(value)

# Convert input to DataFrame
new_farm_df = pd.DataFrame([encoded_input], columns=["soil_type", "pH_level", "NPK_ratio", "water_availability", "crop_preference"])

# Predict the best crop
predicted_crop = model.predict(new_farm_df)
predicted_crop_name = crop_encoder.inverse_transform(predicted_crop)[0]

print(f"Recommended Crop: {predicted_crop_name}")
