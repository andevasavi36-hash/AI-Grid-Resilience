from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from schema import GridInput
from predict import predict_risk

from supabase import create_client
from config import SUPABASE_URL, SUPABASE_KEY


# =========================
# FastAPI App
# =========================

app = FastAPI(
    title="AI Grid Resilience Prediction API",
    description="Predictive Fault Risk Analysis for Power Grid Assets",
    version="1.0"
)


# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Supabase Connection
# =========================

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


# =========================
# Home
# =========================

@app.get("/")
def home():
    return {
        "message": "AI Grid Resilience Prediction API is Running"
    }


# =========================
# Prediction
# =========================

@app.post("/predict")
def prediction(data: GridInput):

    # Convert request to dictionary
    input_data = data.dict()

    # Model Prediction
    result = predict_risk(input_data)

    # Add prediction result
    input_data["Risk_Percentage"] = result

    # Save prediction to Supabase
    try:
        supabase.table("grid_resilience_dataset").insert(
            input_data
        ).execute()
    except Exception as e:
        print("Supabase insert error:", e)

    # Return Prediction
    return {
        "Predicted_Risk_Percentage": round(result, 2)
    }
