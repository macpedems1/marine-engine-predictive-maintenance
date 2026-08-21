"""
Marine Diesel Engine Predictive Maintenance -- Streamlit App
--------------------------------------------------------------
Run with:  streamlit run app.py
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

st.set_page_config(
    page_title="Marine Diesel Engine Health Monitor",
    page_icon="🛢️",
    layout="wide",
)

DATA_PATH = "diesel_engine_maintenance_dataset.csv"  # Update path if inside 'dataset/' folder

SENSOR_OPTIONS = [
    "exhaust_gas_temp_c",
    "exhaust_gas_temp_deviation_c",
    "cooling_water_temp_c",
    "lube_oil_pressure_bar",
    "lube_oil_temp_c",
    "fuel_consumption_rate_kg_h",
    "turbocharger_rpm",
    "vibration_mm_s",
    "lube_oil_fe_ppm",
    "lube_oil_cu_ppm",
    "fuel_air_ratio",
    "injection_timing_deg_btdc",
    "load_percent",
    "rpm",
]


# ---------------------------------------------------------------------------
# DATA & ALL 5 SAVED MODELS LOADING
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
  return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_all_artifacts():
  rf_clf = joblib.load("models/rf_failure_classifier.pkl")
  xgb_clf = joblib.load("models/xgb_failure_classifier.pkl")
  rf_reg = joblib.load("models/rf_rul_regressor.pkl")
  xgb_reg = joblib.load("models/xgb_rul_regressor.pkl")
  iso_forest = joblib.load("models/isolation_forest.pkl")
  scaler = joblib.load("models/feature_scaler.pkl")

  with open("models/feature_list.json", "r") as f:
    model_features = json.load(f)

  return {
      "rf_clf": rf_clf,
      "xgb_clf": xgb_clf,
      "rf_reg": rf_reg,
      "xgb_reg": xgb_reg,
      "iso_forest": iso_forest,
      "scaler": scaler,
      "features": model_features,
  }


df = load_data()
artifacts = load_all_artifacts()

st.title("🛢️ Marine Diesel Engine Health Monitor")
st.caption(
    f"{len(df):,} hourly readings · {df['engine_id'].nunique()} engines · "
    f"{df['failure_event'].sum()} failure events · Predictive maintenance case"
    " study"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Sensor Explorer",
    "🏆 Engine Scorecard",
    "🤖 Model Performance",
    "⚠️ Live Risk Check",
])

# ---------------------------------------------------------------------------
# TAB 1 -- SENSOR EXPLORER
# ---------------------------------------------------------------------------
with tab1:
  st.subheader("Sensor Trend Explorer")
  col1, col2 = st.columns([1, 1])
  with col1:
    engine_choice = st.selectbox(
        "Select engine", sorted(df["engine_id"].unique())
    )
  with col2:
    sensor_choice = st.selectbox(
        "Select sensor",
        SENSOR_OPTIONS,
        index=SENSOR_OPTIONS.index("lube_oil_fe_ppm"),
    )

  engine_df = df[df["engine_id"] == engine_choice].sort_values("operating_hour")

  fig = px.line(
      engine_df,
      x="operating_hour",
      y=sensor_choice,
      title=f"{sensor_choice} over time -- {engine_choice}",
  )
  anomaly_rows = engine_df[engine_df["anomaly_flag"] == 1]
  if len(anomaly_rows) > 0:
    fig.add_vrect(
        x0=anomaly_rows["operating_hour"].min(),
        x1=anomaly_rows["operating_hour"].max(),
        fillcolor="red",
        opacity=0.15,
        line_width=0,
        annotation_text="Degradation window",
        annotation_position="top left",
    )
  fig.update_traces(line_color="#1F3864")
  st.plotly_chart(fig, use_container_width=True)

  if engine_df["failure_event"].sum() > 0:
    fail_hour = engine_df.loc[
        engine_df["failure_event"] == 1, "operating_hour"
    ].iloc[0]
    st.warning(
        f"⚠️ This engine experienced a failure event at operating hour"
        f" {fail_hour}."
    )
  else:
    st.success("✅ This engine completed its run with no failure event.")

# ---------------------------------------------------------------------------
# TAB 2 -- ENGINE SCORECARD
# ---------------------------------------------------------------------------
with tab2:
  st.subheader("Engine Health Scorecard")
  st.caption(
      "Ranked by average lube oil iron content -- the top predictor identified"
      " by the failure model."
  )

  scorecard = (
      df.groupby("engine_id")
      .agg(
          avg_fe_ppm=("lube_oil_fe_ppm", "mean"),
          avg_vibration=("vibration_mm_s", "mean"),
          failed=("failure_event", "max"),
      )
      .reset_index()
      .sort_values("avg_fe_ppm", ascending=False)
  )
  scorecard["status"] = scorecard["failed"].map({1: "Failed", 0: "Healthy"})

  fig2 = px.bar(
      scorecard,
      x="avg_fe_ppm",
      y="engine_id",
      color="status",
      color_discrete_map={"Failed": "#C0392B", "Healthy": "#B8BDC2"},
      orientation="h",
      title="Average Lube Oil Fe (ppm) by Engine",
  )
  fig2.update_layout(yaxis={"categoryorder": "total ascending"}, height=800)
  st.plotly_chart(fig2, use_container_width=True)

  st.dataframe(
      scorecard.style.format(
          {"avg_fe_ppm": "{:.1f}", "avg_vibration": "{:.2f}"}
      ),
      use_container_width=True,
  )

# ---------------------------------------------------------------------------
# TAB 3 -- MODEL PERFORMANCE (ALL 5 MODELS COMPARISON)
# ---------------------------------------------------------------------------
with tab3:
  st.subheader("Model Performance Comparison")

  # Prepare test dataframe for validation metrics evaluation
  work = pd.get_dummies(df, columns=["sea_state"], prefix="sea")
  for col in artifacts["features"]:
    if col not in work.columns:
      work[col] = 0

  X_eval = work[artifacts["features"]]
  y_anom = work["anomaly_flag"]
  y_fail = work["failure_within_48h"]
  y_rul = work["remaining_useful_life_hours"]

  # 1. Isolation Forest Evaluation
  X_scaled = artifacts["scaler"].transform(X_eval)
  anom_preds = (artifacts["iso_forest"].predict(X_scaled) == -1).astype(int)

  st.markdown("**1. Anomaly Detection -- Isolation Forest**")
  c1, c2, c3 = st.columns(3)
  c1.metric("Precision", f"{precision_score(y_anom, anom_preds):.3f}")
  c2.metric("Recall", f"{recall_score(y_anom, anom_preds):.3f}")
  c3.metric("F1 score", f"{f1_score(y_anom, anom_preds):.3f}")

  st.markdown("---")

  # 2. Classifier Comparison: Random Forest vs. XGBoost
  rf_clf_pred = artifacts["rf_clf"].predict(X_eval)
  xgb_clf_pred = artifacts["xgb_clf"].predict(X_eval)
  rf_clf_proba = artifacts["rf_clf"].predict_proba(X_eval)[:, 1]
  xgb_clf_proba = artifacts["xgb_clf"].predict_proba(X_eval)[:, 1]

  clf_summary = pd.DataFrame({
      "Model": ["Random Forest Classifier", "XGBoost Classifier"],
      "Precision": [
          precision_score(y_fail, rf_clf_pred),
          precision_score(y_fail, xgb_clf_pred),
      ],
      "Recall": [
          recall_score(y_fail, rf_clf_pred),
          recall_score(y_fail, xgb_clf_pred),
      ],
      "F1": [f1_score(y_fail, rf_clf_pred), f1_score(y_fail, xgb_clf_pred)],
      "ROC-AUC": [
          roc_auc_score(y_fail, rf_clf_proba),
          roc_auc_score(y_fail, xgb_clf_proba),
      ],
  })

  st.markdown(
      "**2. Failure Classification (within 48 hours) -- Random Forest vs."
      " XGBoost**"
  )
  st.dataframe(
      clf_summary.style.format({
          "Precision": "{:.3f}",
          "Recall": "{:.3f}",
          "F1": "{:.3f}",
          "ROC-AUC": "{:.3f}",
      }),
      use_container_width=True,
  )

  col_roc, col_cm = st.columns(2)
  with col_roc:
    fpr_rf, tpr_rf, _ = roc_curve(y_fail, rf_clf_proba)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_fail, xgb_clf_proba)
    fig_roc = go.Figure()
    fig_roc.add_trace(
        go.Scatter(
            x=fpr_rf, y=tpr_rf, name="Random Forest", line=dict(color="#1F3864")
        )
    )
    fig_roc.add_trace(
        go.Scatter(
            x=fpr_xgb, y=tpr_xgb, name="XGBoost", line=dict(color="#C0790A")
        )
    )
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            name="Random guess",
            line=dict(dash="dash", color="grey"),
        )
    )
    fig_roc.update_layout(
        title="ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    st.plotly_chart(fig_roc, use_container_width=True)

  with col_cm:
    cm = confusion_matrix(y_fail, xgb_clf_pred)
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["No Failure", "Failure"],
        y=["No Failure", "Failure"],
        title="Confusion Matrix -- XGBoost Classifier",
    )
    st.plotly_chart(fig_cm, use_container_width=True)

  st.markdown("---")

  # 3. Regressor Comparison: Random Forest vs. XGBoost
  rf_reg_pred = artifacts["rf_reg"].predict(X_eval)
  xgb_reg_pred = artifacts["xgb_reg"].predict(X_eval)

  reg_summary = pd.DataFrame({
      "Model": ["Random Forest Regressor", "XGBoost Regressor"],
      "MAE (hrs)": [
          mean_absolute_error(y_rul, rf_reg_pred),
          mean_absolute_error(y_rul, xgb_reg_pred),
      ],
      "RMSE (hrs)": [
          mean_squared_error(y_rul, rf_reg_pred) ** 0.5,
          mean_squared_error(y_rul, xgb_reg_pred) ** 0.5,
      ],
      "R2": [r2_score(y_rul, rf_reg_pred), r2_score(y_rul, xgb_reg_pred)],
  })

  st.markdown(
      "**3. Remaining Useful Life (RUL) Regression -- Random Forest vs."
      " XGBoost**"
  )
  st.dataframe(
      reg_summary.style.format(
          {"MAE (hrs)": "{:.1f}", "RMSE (hrs)": "{:.1f}", "R2": "{:.3f}"}
      ),
      use_container_width=True,
  )

  st.markdown("---")
  st.markdown("**4. Feature Importance -- Random Forest Classifier**")
  importances = pd.Series(
      artifacts["rf_clf"].feature_importances_, index=artifacts["features"]
  ).sort_values(ascending=True)
  fig_imp = px.bar(
      importances.tail(12),
      orientation="h",
      labels={"value": "Importance", "index": "Feature"},
      title="Top 12 Feature Importances",
  )
  fig_imp.update_traces(marker_color="#1F3864")
  st.plotly_chart(fig_imp, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 4 -- LIVE RISK CHECK
# ---------------------------------------------------------------------------
with tab4:
  st.subheader("Live Risk Check")
  st.caption(
      "Pick an engine and an operating hour to see the model's failure-risk"
      " probability and predicted remaining useful life at that point in time."
  )

  engine_choice2 = st.selectbox(
      "Engine", sorted(df["engine_id"].unique()), key="risk_engine"
  )
  engine_rows = df[df["engine_id"] == engine_choice2].sort_values(
      "operating_hour"
  )
  hour_choice = st.slider(
      "Operating hour",
      int(engine_rows["operating_hour"].min()),
      int(engine_rows["operating_hour"].max()),
      int(engine_rows["operating_hour"].max()),
  )

  row = engine_rows[engine_rows["operating_hour"] == hour_choice].iloc[0:1].copy()
  row = pd.get_dummies(row, columns=["sea_state"], prefix="sea")

  for f in artifacts["features"]:
    if f not in row.columns:
      row[f] = 0
  row = row[artifacts["features"]]

  fail_proba = artifacts["xgb_clf"].predict_proba(row)[0, 1]
  rul_pred = max(0, artifacts["rf_reg"].predict(row)[0])
  has_failed_previously = (
      engine_rows[engine_rows["operating_hour"] <= hour_choice][
          "failure_event"
      ].sum()
      > 0
  )

  c1, c2 = st.columns(2)
  with c1:
    st.metric("Failure risk (next 48 hrs)", f"{fail_proba * 100:.1f}%")
    if rul_pred <= 1 or has_failed_previously:
      st.error(
          "🚨 ENGINE OUT OF SERVICE — Operating past maximum safe service limit."
      )
    elif fail_proba > 0.5:
      st.error(
          "High risk -- recommend scheduling inspection at next port call."
      )
    elif fail_proba > 0.2:
      st.warning(
          "Medium risk -- monitor closely, consider inspection at next port"
          " call."
      )
    else:
      st.success("Low risk.")
  with c2:
    st.metric("Predicted remaining useful life", f"{rul_pred:.0f} hours")

  st.markdown("**Sensor readings at this point:**")

  actual_row = engine_rows[engine_rows["operating_hour"] == hour_choice][[
      "operating_hour",
      "exhaust_gas_temp_c",
      "exhaust_gas_temp_deviation_c",
      "vibration_mm_s",
      "lube_oil_fe_ppm",
      "lube_oil_pressure_bar",
      "turbocharger_rpm",
  ]].copy()

  actual_row.columns = [
      "Operating Hour (hrs)",
      "Exhaust Temp (°C)",
      "Temp Deviation (°C)",
      "Vibration (mm/s)",
      "Lube Oil Fe (ppm)",
      "Lube Oil Pressure (bar)",
      "Turbocharger Speed (RPM)",
  ]

  st.dataframe(actual_row, use_container_width=True, hide_index=True)