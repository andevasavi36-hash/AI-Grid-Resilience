# ⚡ AI-Powered Grid Resilience Intelligence Platform

An AI-powered predictive analytics platform designed to assess **power grid equipment fault risk** using machine learning.

The system analyzes equipment, load, temperature, weather, stability, and maintenance-related parameters to generate a **Risk Percentage** and classify the equipment into different risk categories.

---

## 🚀 Project Overview

Power grids operate under continuously changing conditions such as fluctuating loads, equipment aging, temperature variations, and weather conditions.

Traditional monitoring systems often identify faults after they occur. This project aims to provide **predictive fault risk analysis** so that potential high-risk equipment can be identified earlier and preventive maintenance can be planned.

### 🎯 Objective

* Predict equipment fault risk percentage
* Identify low, medium, high, and critical-risk conditions
* Support predictive maintenance decisions
* Provide an easy-to-use web interface
* Integrate ML prediction with a FastAPI backend
* Store and manage project data using Supabase

---

## 🧠 Machine Learning

The project uses supervised machine learning for predicting equipment risk.

### Models Evaluated

* Linear Regression
* KNN Regressor
* Decision Tree Regressor
* Random Forest Regressor
* XGBoost Regressor
* Support Vector Regression (SVR)

### Selected Model

**Random Forest Regressor**

Random Forest was selected based on its prediction performance and ability to model nonlinear relationships between grid equipment parameters and risk.

---

## 📊 Input Features

The prediction system uses important grid and equipment parameters such as:

| Feature                 | Description                       |
| ----------------------- | --------------------------------- |
| Equipment Age           | Age of the equipment in years     |
| Load Percentage         | Current equipment load            |
| Transformer Temperature | Transformer operating temperature |
| Grid Stability Score    | Stability score of the grid       |
| Weather Temperature     | Current environmental temperature |
| Humidity                | Environmental humidity            |
| Weather Condition       | Current weather condition         |
| Maintenance Status      | Current maintenance condition     |

The user-friendly frontend focuses on the most important parameters instead of exposing every raw dataset column.

---

## 📈 Risk Classification

The predicted risk percentage is classified as:

| Risk Percentage | Risk Level       |
| --------------: | ---------------- |
|           0–30% | 🟢 Low Risk      |
|          31–60% | 🟡 Medium Risk   |
|          61–80% | 🟠 High Risk     |
|         81–100% | 🔴 Critical Risk |

> **Note:** Risk percentage is an ML model prediction and should be treated as an analytical estimate, not a guaranteed outcome.

---

## 🏗️ System Architecture

```text
                  ┌─────────────────────┐
                  │     User Input      │
                  │    Streamlit UI     │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │    FastAPI Backend  │
                  │   Prediction API    │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Preprocessing       │
                  │ Encoding + Scaling  │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Random Forest Model │
                  │    Risk Prediction  │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Risk Percentage +   │
                  │ Risk Classification │
                  └─────────────────────┘
```

---

## 🛠️ Tech Stack

### Programming

* Python

### Machine Learning

* Scikit-learn
* Random Forest
* XGBoost
* NumPy
* Pandas

### Backend

* FastAPI
* Uvicorn
* Pydantic

### Frontend

* Streamlit

### Database

* Supabase

### Model Serialization

* Joblib

### Development Tools

* VS Code
* Jupyter Notebook
* Git
* GitHub

---

## 📁 Project Structure

```text
AI-Grid-Resilience/
│
├── APP/
│   │
│   ├── backend/
│   │   ├── main.py
│   │   ├── predict.py
│   │   └── ...
│   │
│   └── frontend/
│       └── app.py
│
├── Pickles/
│   ├── random_forest_model.pkl
│   ├── standard_scaler.pkl
│   └── onehot_encoder.pkl
│
├── dataset/
│   └── grid_resilience_dataset.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   └── Model_Training.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AI-Grid-Resilience
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Backend

Navigate to the backend directory:

```bash
cd APP/backend
```

Start FastAPI:

```bash
uvicorn main:app --reload
```

Backend will be available at:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 🖥️ Run the Frontend

Open another terminal and activate the virtual environment.

Navigate to the frontend:

```bash
cd APP/frontend
```

Run Streamlit:

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

---

## 🔌 API Workflow

The prediction request follows this workflow:

```text
User Input
    ↓
Streamlit
    ↓
FastAPI POST Request
    ↓
Input Validation
    ↓
Feature Preprocessing
    ↓
Encoding
    ↓
Scaling
    ↓
Random Forest Prediction
    ↓
Risk Percentage
    ↓
Risk Category
    ↓
Streamlit Result
```

---

## 🗄️ Supabase Integration

Supabase is used for database functionality and storing application-related prediction data.

The application can be extended to store:

* Equipment information
* Prediction results
* Risk percentages
* Risk categories
* Prediction timestamps
* Maintenance information

Environment variables should be used for sensitive credentials.

Example:

```text
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

**Never commit API keys, passwords, or other credentials to GitHub.**

---

## 📌 Key Features

### 🏠 Home

Provides an overview of the AI Grid Resilience platform.

### 📊 Project Overview

Displays:

* Dataset information
* Machine learning models
* Model performance
* Selected model
* Project methodology

### 🔮 Predictions

Allows users to enter important equipment and grid parameters and receive:

* Risk Percentage
* Risk Level
* Prediction interpretation

---

## 🎯 Use Cases

This platform can be useful for:

* Predictive maintenance
* Transformer monitoring
* Grid equipment risk assessment
* Fault prevention
* Power distribution monitoring
* Energy infrastructure analytics

---

## 🔮 Future Improvements

* Real-time IoT sensor integration
* Live grid monitoring dashboard
* Automated maintenance alerts
* Time-series forecasting
* Explainable AI using SHAP
* Historical risk trend analysis
* Real-time weather API integration
* Automated anomaly detection
* Cloud-based model retraining
* Role-based monitoring dashboards

---

## ⚠️ Disclaimer

This application provides **machine-learning-based risk estimates** for analytical and demonstration purposes.

The predictions are not guaranteed to represent actual equipment failures and should not be used as the sole basis for critical power-grid operational decisions.

---

## 👨‍💻 Author

**Vasavi**

Aspiring Data Scientist | Machine Learning | Python | Data Analytics

---

## ⭐ Project Highlights

* End-to-end Machine Learning project
* Data preprocessing and feature engineering
* Multiple ML models evaluated
* Random Forest-based risk prediction
* FastAPI backend
* Streamlit frontend
* Supabase integration
* Deployable ML application
* Predictive maintenance use case

---

## 📜 License

This project is intended for educational, portfolio, and demonstration purposes.
