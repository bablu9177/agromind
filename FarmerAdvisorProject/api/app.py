import joblib
import pandas as pd
from fastapi import FastAPI

# Load the trained model and encoders
model = joblib.load(r"C:\Users\kanth\OneDrive\Desktop\FarmerAdvisorProject\models/farmer_advisor_model.pkl")
label_encoders = joblib.load(r"C:\Users\kanth\OneDrive\Desktop\FarmerAdvisorProject\models/label_encoders.pkl")
y_encoder = joblib.load(r"C:\Users\kanth\OneDrive\Desktop\FarmerAdvisorProject\models/y_encoder.pkl")

# Initialize FastAPI
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Farmer Advisor AI - Crop Recommendation API"}

@app.post("/predict_crop/")
def predict_crop(soil_type: str, pH_level: float, NPK_ratio: str, water_availability: str, crop_preference: str):
    # Prepare input data
    input_data = [soil_type, pH_level, NPK_ratio, water_availability, crop_preference]
    
    # Encode categorical variables
    encoded_input = []
    for key, value in zip(["Soil_Type", "NPK_Ratio", "Water_Availability", "Crop_Preference"], input_data):
        if key in label_encoders:
            encoded_input.append(label_encoders[key].transform([value])[0])
        else:
            encoded_input.append(value)
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([encoded_input], columns=["Soil_Type", "pH_Level", "NPK_Ratio", "Water_Availability", "Crop_Preference"])

    # Predict the best crop
    prediction = model.predict(input_df)

    # Decode the prediction
    predicted_label = y_encoder.inverse_transform(prediction)[0]

    return {"recommended_crop": predicted_label}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
