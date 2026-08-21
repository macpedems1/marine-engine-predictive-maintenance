"""
Marine Diesel Engine Predictive Maintenance -- Streamlit App

Run with:  streamlit run app.py

Tabs:
  1. Sensor Explorer      -- pick an engine + sensor, see the trend, anomaly window highlighted
  2. Engine Scorecard      -- rank all engines by risk, healthy vs. failed
  3. Model Performance     -- train Random Forest + XGBoost live, compare metrics, ROC, confusion matrix, feature importance
  4. Live Risk Check       -- pick an engine's latest reading, get a failure-risk probability + predicted RUL
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
)
from xgboost import XGBClassifier, XGBRegressor

st.set_page_config(
    page_title="Marine Diesel Engine Health Monitor",
    page_icon="🛢️",
    layout="wide",
)

DATA_PATH = "dataset/diesel_engine_maintenance_dataset.csv"
RANDOM_STATE = 42

FEATURES = [
    "operating_hour", "hours_since_last_overhaul", "days_since_last_maintenance",
    "load_percent", "rpm", "exhaust_gas_temp_c", "exhaust_gas_temp_deviation_c",
    "cooling_water_temp_c", "lube_oil_pressure_bar", "lube_oil_temp_c",
    "fuel_consumption_rate_kg_h", "turbocharger_rpm", "vibration_mm_s",
    "lube_oil_fe_ppm", "lube_oil_cu_ppm", "ambient_temp_c",
    "fuel_air_ratio", "injection_timing_deg_btdc",
]

SENSOR_OPTIONS = [
    "exhaust_gas_temp_c", "exhaust_gas_temp_deviation_c", "cooling_water_temp_c",
    "lube_oil_pressure_bar", "lube_oil_temp_c", "fuel_consumption_rate_kg_h",
    "turbocharger_rpm", "vibration_mm_s", "lube_oil_fe_ppm", "lube_oil_cu_ppm",
    "fuel_air_ratio", "injection_timing_deg_btdc", "load_percent", "rpm",
]


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


# ---------------------------------------------------------------------------
# MODEL TRAINING (cached so it only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def train_models(df):
    work = pd.get_dummies(df, columns=["sea_state"], prefix="sea")
    sea_cols = [c for c in work.columns if c.startswith("sea_")]
    features = FEATURES + sea_cols

    X = work[features]
    y_anomaly = work["anomaly_flag"]
    y_failure = work["failure_within_48h"]
    y_rul = work["remaining_useful_life_hours"]
    groups = work["engine_id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y_failure, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_anom_train, y_anom_test = y_anomaly.iloc[train_idx], y_anomaly.iloc[test_idx]
    y_fail_train, y_fail_test = y_failure.iloc[train_idx], y_failure.iloc[test_idx]
    y_rul_train, y_rul_test = y_rul.iloc[train_idx], y_rul.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Isolation Forest (anomaly detection) ---
    iso = IsolationForest(n_estimators=200, contamination=0.06, random_state=RANDOM_STATE)
    iso.fit(X_train_scaled)
    anom_pred = (iso.predict(X_test_scaled) == -1).astype(int)
    iso_metrics = {
        "Precision": precision_score(y_anom_test, anom_pred),
        "Recall": recall_score(y_anom_test, anom_pred),
        "F1": f1_score(y_anom_test, anom_pred),
    }

    # --- Random Forest classifier ---
    rf_clf = RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced", random_state=RANDOM_STATE
    )
    rf_clf.fit(X_train, y_fail_train)
    rf_proba = rf_clf.predict_proba(X_test)[:, 1]
    rf_pred = rf_clf.predict(X_test)

    # --- XGBoost classifier ---
    pos = y_fail_train.sum()
    neg = len(y_fail_train) - pos
    xgb_clf = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=neg / max(pos, 1), eval_metric="logloss", random_state=RANDOM_STATE,
    )
    xgb_clf.fit(X_train, y_fail_train)
    xgb_proba = xgb_clf.predict_proba(X_test)[:, 1]
    xgb_pred = xgb_clf.predict(X_test)

    clf_summary = pd.DataFrame({
        "Model": ["Random Forest", "XGBoost"],
        "Precision": [precision_score(y_fail_test, rf_pred), precision_score(y_fail_test, xgb_pred)],
        "Recall": [recall_score(y_fail_test, rf_pred), recall_score(y_fail_test, xgb_pred)],
        "F1": [f1_score(y_fail_test, rf_pred), f1_score(y_fail_test, xgb_pred)],
        "ROC-AUC": [roc_auc_score(y_fail_test, rf_proba), roc_auc_score(y_fail_test, xgb_proba)],
    })

    # --- Random Forest regressor (RUL) ---
    rf_reg = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=RANDOM_STATE)
    rf_reg.fit(X_train, y_rul_train)
    rf_reg_pred = rf_reg.predict(X_test)

    # --- XGBoost regressor (RUL) ---
    xgb_reg = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=RANDOM_STATE)
    xgb_reg.fit(X_train, y_rul_train)
    xgb_reg_pred = xgb_reg.predict(X_test)

    reg_summary = pd.DataFrame({
        "Model": ["Random Forest", "XGBoost"],
        "MAE (hrs)": [mean_absolute_error(y_rul_test, rf_reg_pred), mean_absolute_error(y_rul_test, xgb_reg_pred)],
        "RMSE (hrs)": [
            mean_squared_error(y_rul_test, rf_reg_pred) ** 0.5,
            mean_squared_error(y_rul_test, xgb_reg_pred) ** 0.5,
        ],
        "R2": [r2_score(y_rul_test, rf_reg_pred), r2_score(y_rul_test, xgb_reg_pred)],
    })

    fpr_rf, tpr_rf, _ = roc_curve(y_fail_test, rf_proba)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_fail_test, xgb_proba)

    return {
        "features": features,
        "scaler": scaler,
        "iso_forest": iso,
        "iso_metrics": iso_metrics,
        "rf_clf": rf_clf,
        "xgb_clf": xgb_clf,
        "clf_summary": clf_summary,
        "rf_reg": rf_reg,
        "xgb_reg": xgb_reg,
        "reg_summary": reg_summary,
        "y_fail_test": y_fail_test,
        "rf_pred": rf_pred,
        "xgb_pred": xgb_pred,
        "roc_rf": (fpr_rf, tpr_rf),
        "roc_xgb": (fpr_xgb, tpr_xgb),
    }


# ---------------------------------------------------------------------------
# LOAD + TRAIN
# ---------------------------------------------------------------------------
df = load_data()
models = train_models(df)

st.title("🛢️ Marine Diesel Engine Health Monitor")
st.caption(
    f"{len(df):,} hourly readings · {df['engine_id'].nunique()} engines · "
    f"{df['failure_event'].sum()} failure events · Predictive maintenance case study"
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Sensor Explorer", "🏆 Engine Scorecard", "🤖 Model Performance", "⚠️ Live Risk Check"]
)

# ---------------------------------------------------------------------------
# TAB 1 -- SENSOR EXPLORER
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Sensor Trend Explorer")
    col1, col2 = st.columns([1, 1])
    with col1:
        engine_choice = st.selectbox("Select engine", sorted(df["engine_id"].unique()))
    with col2:
        sensor_choice = st.selectbox("Select sensor", SENSOR_OPTIONS, index=SENSOR_OPTIONS.index("lube_oil_fe_ppm"))

    engine_df = df[df["engine_id"] == engine_choice].sort_values("operating_hour")

    fig = px.line(
        engine_df, x="operating_hour", y=sensor_choice,
        title=f"{sensor_choice} over time -- {engine_choice}",
    )
    # highlight the anomaly window
    anomaly_rows = engine_df[engine_df["anomaly_flag"] == 1]
    if len(anomaly_rows) > 0:
        fig.add_vrect(
            x0=anomaly_rows["operating_hour"].min(), x1=anomaly_rows["operating_hour"].max(),
            fillcolor="red", opacity=0.15, line_width=0,
            annotation_text="Degradation window", annotation_position="top left",
        )
    fig.update_traces(line_color="#1F3864")
    st.plotly_chart(fig, use_container_width=True)

    if engine_df["failure_event"].sum() > 0:
        fail_hour = engine_df.loc[engine_df["failure_event"] == 1, "operating_hour"].iloc[0]
        st.warning(f"⚠️ This engine experienced a failure event at operating hour {fail_hour}.")
    else:
        st.success("✅ This engine completed its run with no failure event.")

# ---------------------------------------------------------------------------
# TAB 2 -- ENGINE SCORECARD
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Engine Health Scorecard")
    st.caption("Ranked by average lube oil iron content -- the top predictor identified by the failure model.")

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
        scorecard, x="avg_fe_ppm", y="engine_id", color="status",
        color_discrete_map={"Failed": "#C0392B", "Healthy": "#B8BDC2"},
        orientation="h", title="Average Lube Oil Fe (ppm) by Engine",
    )
    fig2.update_layout(yaxis={"categoryorder": "total ascending"}, height=800)
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(scorecard.style.format({"avg_fe_ppm": "{:.1f}", "avg_vibration": "{:.2f}"}), use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3 -- MODEL PERFORMANCE
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Model Performance")

    st.markdown("**Anomaly Detection -- Isolation Forest**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Precision", f"{models['iso_metrics']['Precision']:.3f}")
    c2.metric("Recall", f"{models['iso_metrics']['Recall']:.3f}")
    c3.metric("F1 score", f"{models['iso_metrics']['F1']:.3f}")

    st.markdown("---")
    st.markdown("**Failure Classification (within 48 hours) -- Random Forest vs. XGBoost**")
    st.dataframe(models["clf_summary"].style.format({
        "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}", "ROC-AUC": "{:.3f}"
    }), use_container_width=True)

    col_roc, col_cm = st.columns(2)
    with col_roc:
        fpr_rf, tpr_rf = models["roc_rf"]
        fpr_xgb, tpr_xgb = models["roc_xgb"]
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, name="Random Forest", line=dict(color="#1F3864")))
        fig_roc.add_trace(go.Scatter(x=fpr_xgb, y=tpr_xgb, name="XGBoost", line=dict(color="#C0790A")))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random guess", line=dict(dash="dash", color="grey")))
        fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)
    with col_cm:
        cm = confusion_matrix(models["y_fail_test"], models["xgb_pred"])
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["No Failure", "Failure"], y=["No Failure", "Failure"],
            title="Confusion Matrix -- XGBoost",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")
    st.markdown("**Remaining Useful Life (RUL) Regression -- Random Forest vs. XGBoost**")
    st.dataframe(models["reg_summary"].style.format({
        "MAE (hrs)": "{:.1f}", "RMSE (hrs)": "{:.1f}", "R2": "{:.3f}"
    }), use_container_width=True)

    st.markdown("---")
    st.markdown("**Feature Importance -- Random Forest Failure Classifier**")
    importances = pd.Series(models["rf_clf"].feature_importances_, index=models["features"]).sort_values(ascending=True)
    fig_imp = px.bar(
        importances.tail(12), orientation="h",
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
    st.caption("Pick an engine and an operating hour to see the model's failure-risk probability and predicted remaining useful life at that point in time.")

    engine_choice2 = st.selectbox("Engine", sorted(df["engine_id"].unique()), key="risk_engine")
    engine_rows = df[df["engine_id"] == engine_choice2].sort_values("operating_hour")
    hour_choice = st.slider(
        "Operating hour",
        int(engine_rows["operating_hour"].min()),
        int(engine_rows["operating_hour"].max()),
        int(engine_rows["operating_hour"].max()),
    )

    row = engine_rows[engine_rows["operating_hour"] == hour_choice].iloc[0:1].copy()
    row = pd.get_dummies(row, columns=["sea_state"], prefix="sea")
    for f in models["features"]:
        if f not in row.columns:
            row[f] = 0
    row = row[models["features"]]

    fail_proba = models["xgb_clf"].predict_proba(row)[0, 1]
    rul_pred = max(0, models["rf_reg"].predict(row)[0])
    has_failed_previously = engine_rows[engine_rows["operating_hour"] <= hour_choice]["failure_event"].sum() > 0

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Failure risk (next 48 hrs)", f"{fail_proba * 100:.1f}%")
        if rul_pred <= 1 or has_failed_previously:
            st.error("🚨 ENGINE OUT OF SERVICE — Operating past maximum safe service limit.")
        elif fail_proba > 0.5:
            st.error("High risk -- recommend scheduling inspection at next port call.")
        elif fail_proba > 0.2:
            st.warning("Medium risk -- monitor closely, consider inspection at next port call.")
        else:
            st.success("Low risk.")
    with c2:
        st.metric("Predicted remaining useful life", f"{rul_pred:.0f} hours")

    # st.markdown("**Sensor readings at this point:**")
    # display_cols = [
    #     "exhaust_gas_temp_c", "exhaust_gas_temp_deviation_c", "vibration_mm_s",
    #     "lube_oil_fe_ppm", "lube_oil_pressure_bar", "turbocharger_rpm",
    # ]
    # actual_row = engine_rows[engine_rows["operating_hour"] == hour_choice][display_cols]
    # st.dataframe(actual_row, use_container_width=True)

    # NEW CODE WITH UNITS & RENAME
    st.markdown("**Sensor readings at this point:**")

    actual_row = engine_rows[engine_rows["operating_hour"] == hour_choice][[
        "operating_hour", "exhaust_gas_temp_c", "exhaust_gas_temp_deviation_c", 
        "vibration_mm_s", "lube_oil_fe_ppm", "lube_oil_pressure_bar", "turbocharger_rpm"
    ]].copy()

    # Rename columns to show human-readable labels with clear units
    actual_row.columns = [
        "Operating Hour (hrs)", "Exhaust Temp (°C)", "Temp Deviation (°C)", 
        "Vibration (mm/s)", "Lube Oil Fe (ppm)", "Lube Oil Pressure (bar)", "Turbocharger Speed (RPM)"
    ]

    # Display as clean table without index column
    st.dataframe(actual_row, use_container_width=True, hide_index=True)

