import joblib
import pandas as pd
import numpy as np
from datetime import datetime


# =========================
# Load Model Files
# =========================

model = joblib.load("../Pickles/random_forest_model.pkl")
scaler = joblib.load("../Pickles/standard_scaler.pkl")
encoder = joblib.load("../Pickles/onehot_encoder.pkl")



# =========================
# Prediction Function
# =========================

def predict_risk(data: dict):

    now = datetime.now()


    # Default values

    input_data = {

        "Region": "north",
        "Equipment_Id": "EQ001",
        "Switch_Status": "ON",

        "Hour": now.hour,
        "Day": now.day,
        "Month": now.month,
        "Dayofweek": now.weekday(),

        "Equipment_Capacity_Kva": 500,
        "Equipment_Age_Years": 5,

        "Voltage_V": 230,
        "Voltage_Pu": 1.0,

        "Current_A": 100,
        "Frequency_Hz": 50,

        "Active_Power_Flow_Kw": 120,
        "Reactive_Power_Flow_Kvar": 30,

        "Power_Factor": 0.95,

        "P_Load_Kw": 100,
        "Q_Load_Kvar": 20,

        "Der_P_Output_Kw": 0,
        "Der_Q_Output_Kvar": 0,

        "Line_Resistance_Ohm": 0.5,
        "Line_Reactance_Ohm": 0.3,

        "Regional_Load_Mw": 100,
        "Load_Percentage": 50,

        "Transformer_Temperature_C": 45,
        "Oil_Temperature_C": 45,

        "Weather_Condition": "sunny",
        "Weather_Temperature_C": 30,
        "Humidity_Pct": 50,
        "Rainfall_Mm": 0,
        "Wind_Speed_Kmh": 10,
        "Atmospheric_Pressure_Hpa": 1013,

        "Grid_Stability_Score": 90,

        "Maintenance_Status": "good",
        "Component_Health": "good",
        "Fault_Type": "none",

        "Fault_Duration_Hrs": 0,
        "Downtime_Hrs": 0,

        "Year": now.year
    }



    # User values replace

    for key,value in data.items():

        if value is not None:

            input_data[key] = value



    df = pd.DataFrame([input_data])



    # =========================
    # Encoding
    # =========================

    cat_cols = list(encoder.feature_names_in_)



    for col in cat_cols:

        if col not in df.columns:

            index = cat_cols.index(col)

            df[col] = encoder.categories_[index][0]



    # Handle unknown categories

    for i,col in enumerate(cat_cols):

        if df.loc[0,col] not in encoder.categories_[i]:

            df.loc[0,col] = encoder.categories_[i][0]



    encoded = encoder.transform(df[cat_cols])


    if hasattr(encoded,"toarray"):

        encoded = encoded.toarray()



    # =========================
    # Scaling
    # =========================

    scale_cols = list(scaler.feature_names_in_)


    scale_df = pd.DataFrame()



    for col in scale_cols:


        if col in df.columns:


            value = df[col].iloc[0]


            if isinstance(value,str):

                value = 0


            scale_df[col] = [value]


        else:

            scale_df[col] = [0]



    scale_df = scale_df.astype(float)



    scaled = scaler.transform(scale_df)



    # =========================
    # Final Input
    # =========================

    X = scaled



    # Safety check

    if X.shape[1] != model.n_features_in_:

        raise ValueError(
            f"Feature mismatch: Model expects {model.n_features_in_}, got {X.shape[1]}"
        )



    prediction = model.predict(X)


    return float(prediction[0])





# =========================
# Testing
# =========================

if __name__ == "__main__":


    sample = {

        "Region":"north",

        "Weather_Condition":"rainy",

        "Equipment_Age_Years":15,

        "Load_Percentage":85,

        "Transformer_Temperature_C":90,

        "Maintenance_Status":"poor"

    }


    result = predict_risk(sample)


    print("Risk Percentage:",result)