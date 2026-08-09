from fastapi import FastAPI
from database.schema import GridInput
from database.database import supabase
from predict import predict_risk

app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "AI Grid Resilience Prediction API is Running"
    }


@app.post("/predict")
def prediction(data: GridInput):

    # Convert request to dictionary
    input_data = data.dict()

    # Model Prediction
    result = predict_risk(input_data)

    # Save into Supabase
    input_data["Risk_Percentage"] = result

    supabase.table("grid_resilience_dataset").insert(
        input_data
    ).execute()

    # Return Prediction
    return {
        "Predicted_Risk_Percentage": round(result, 2)
    }