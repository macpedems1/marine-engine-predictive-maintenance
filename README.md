# 🚢 Marine Diesel Engine Predictive Maintenance & Health Dashboard
End-to-end predictive maintenance dashboard for marine diesel engine health monitoring using Streamlit, XGBoost, and Random Forest.


An end-to-end Machine Learning web application built with Python and Streamlit to monitor 40 marine diesel engine units. The system predicts 48-hour failure risks using **XGBoost** and estimates Remaining Useful Life (RUL) in hours using a **Random Forest Regressor**.

---

## 📌 Project Overview

Unplanned mechanical failures at sea lead to severe operational downtime, costly emergency repairs, and safety hazards. This dashboard transitions fleet operations from reactive maintenance to **predictive asset integrity management** by continuously processing engine sensor streams.

### Key Capabilities:
* **Multi-Model Predictive Engine:** Classifies short-term failure probability (48-hr window) and predicts long-term remaining useful life.
* **Interactive Sensor Inspection:** Explores historical operating hours across individual engine IDs (`ENG-01` to `ENG-40`).
* **Operational Risk Alerts:** Displays human-readable alert statuses (`NORMAL`, `ELEVATED RISK`, `HIGH RISK`, `ENGINE OUT OF SERVICE`) paired with actionable maintenance recommendations for marine engineers.
* **Formatted Engineering Metrics:** Converts raw database variables into industry-standard UI metrics with clear physical units (°C, mm/s, ppm, bar, RPM).

---

## 🛠️ Technology Stack

* **Frontend Dashboard:** Streamlit
* **Machine Learning:** Scikit-Learn, XGBoost, Joblib
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Language & Runtime:** Python 3.10+

---

## ⚙️ Key Features Analyzed

The predictive models process real-time telemetry inputs including:
* **Thermal Metrics:** Exhaust gas temperature (°C) & temperature deviation (°C)
* **Vibration & Mechanical Wear:** Overall vibration (mm/s), turbocharger speed (RPM)
* **Lube Oil Analysis:** Iron particulate content (`Lube Oil Fe` in ppm) & oil pressure (bar)
* **Environmental Context:** Sea state conditions (e.g., Moderate, Heavy)

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/marine-engine-predictive-maintenance.git](https://github.com/YOUR_USERNAME/marine-engine-predictive-maintenance.git)
cd marine-engine-predictive-maintenance
