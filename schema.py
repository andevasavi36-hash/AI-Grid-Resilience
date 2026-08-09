from pydantic import BaseModel
from typing import Optional


class GridInput(BaseModel):

    # User Inputs
    Equipment_Age_Years: float
    Load_Percentage: float
    Transformer_Temperature_C: float
    Grid_Stability_Score: float
    Weather_Temperature_C: float
    Humidity_Pct: float

    # Optional fields (backend fills these)
    Region: Optional[str] = None
    Equipment_Id: Optional[str] = None
    Hour: Optional[float] = None
    Day: Optional[float] = None
    Month: Optional[float] = None
    Dayofweek: Optional[float] = None
    Equipment_Capacity_Kva: Optional[float] = None
    Voltage_V: Optional[float] = None
    Voltage_Pu: Optional[float] = None
    Current_A: Optional[float] = None
    Frequency_Hz: Optional[float] = None
    Active_Power_Flow_Kw: Optional[float] = None
    Reactive_Power_Flow_Kvar: Optional[float] = None
    Power_Factor: Optional[float] = None
    P_Load_Kw: Optional[float] = None
    Q_Load_Kvar: Optional[float] = None
    Der_P_Output_Kw: Optional[float] = None
    Der_Q_Output_Kvar: Optional[float] = None
    Line_Resistance_Ohm: Optional[float] = None
    Line_Reactance_Ohm: Optional[float] = None
    Switch_Status: Optional[str] = None
    Regional_Load_Mw: Optional[float] = None
    Oil_Temperature_C: Optional[float] = None
    Weather_Condition: Optional[str] = None
    Rainfall_Mm: Optional[float] = None
    Wind_Speed_Kmh: Optional[float] = None
    Atmospheric_Pressure_Hpa: Optional[float] = None
    Maintenance_Status: Optional[str] = None
    Component_Health: Optional[str] = None
    Fault_Type: Optional[str] = None
    Fault_Duration_Hrs: Optional[float] = None
    Downtime_Hrs: Optional[float] = None
    Year: Optional[float] = None