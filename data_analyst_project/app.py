"""
Healthcare No-Show Prediction — Streamlit Frontend
=====================================================
Run:   streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, confusion_matrix

from ml_backend import (
    load_data, engineer_features, get_X_y, FEATURE_COLS,
    PLOTS_DIR, ARTIFACTS_PATH, DATA_PATH
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare No-Show Predictor",
    page_icon="🏥",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-title { font-size: 13px; color: #57606a; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: 700; color: #1f2328; }
    .metric-sub   { font-size: 12px; color: #57606a; }
    h1 { color: #1f2328; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🏥 No-Show Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Overview & EDA", "🤖 Model Performance", "🔮 Predict No-Show"]
)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_processed():
    df_raw = load_data(DATA_PATH)
    df     = engineer_features(df_raw.copy())
    return df_raw, df


@st.cache_resource
def load_artifacts():
    if not os.path.exists(ARTIFACTS_PATH):
        st.error("Model artifacts not found. Run `python ml_backend.py` first.")
        st.stop()
    return joblib.load(ARTIFACTS_PATH)


def metric_card(title, value, sub=""):
    return f"""
    <div class="metric-card">
      <div class="metric-title">{title}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-sub">{sub}</div>
    </div>"""


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — Overview & EDA
# ════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview & EDA":
    st.title("📊 Healthcare No-Show — Exploratory Data Analysis")

    df_raw, df = load_processed()

    # Top KPI row
    total      = len(df_raw)
    noshow_n   = int(df["no_show"].sum())
    noshow_pct = noshow_n / total * 100
    avg_wait   = df["wait_days"].mean()
    sms_rate   = df["sms_received"].mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Total Appointments", f"{total:,}"), unsafe_allow_html=True)
    c2.markdown(metric_card("No-Shows", f"{noshow_n:,}", f"{noshow_pct:.1f}% of total"), unsafe_allow_html=True)
    c3.markdown(metric_card("Avg Wait Days", f"{avg_wait:.1f}"), unsafe_allow_html=True)
    c4.markdown(metric_card("SMS Reminder Rate", f"{sms_rate:.1f}%"), unsafe_allow_html=True)

    st.markdown("---")

    # Class distribution
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Class Distribution")
        vc = df["no_show"].value_counts().rename({0: "Showed Up", 1: "No-Show"})
        fig = px.bar(x=vc.index, y=vc.values,
                     color=vc.index,
                     color_discrete_map={"Showed Up": "#3b82d4", "No-Show": "#e05c5c"},
                     labels={"x": "Outcome", "y": "Count"},
                     text=vc.values)
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("No-Show by Gender")
        grp = df_raw.copy()
        grp["no_show"] = df["no_show"]
        g_rate = grp.groupby("gender")["no_show"].mean().reset_index()
        g_rate.columns = ["Gender", "No-Show Rate"]
        g_rate["Gender"] = g_rate["Gender"].map({"F": "Female", "M": "Male"})
        fig2 = px.bar(g_rate, x="Gender", y="No-Show Rate",
                      color="Gender",
                      color_discrete_map={"Female": "#7c5cd8", "Male": "#3b82d4"},
                      text=g_rate["No-Show Rate"].apply(lambda x: f"{x*100:.1f}%"))
        fig2.update_traces(textposition="outside")
        fig2.update_layout(showlegend=False, height=350, yaxis_tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

    # Age distribution
    st.subheader("Age Distribution by Outcome")
    df_plot = df.copy()
    df_plot["Outcome"] = df_plot["no_show"].map({0: "Showed Up", 1: "No-Show"})
    fig3 = px.histogram(df_plot, x="age", color="Outcome", nbins=50, barmode="overlay",
                        opacity=0.7,
                        color_discrete_map={"Showed Up": "#3b82d4", "No-Show": "#e05c5c"},
                        labels={"age": "Age"})
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)

    # Wait days vs outcome
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Wait Days Distribution")
        fig4 = px.box(df_plot, x="Outcome", y="wait_days",
                      color="Outcome",
                      color_discrete_map={"Showed Up": "#3b82d4", "No-Show": "#e05c5c"})
        fig4.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig4, use_container_width=True)

    with col4:
        st.subheader("SMS Reminder Impact")
        sms_df = df.groupby("sms_received")["no_show"].mean().reset_index()
        sms_df["SMS"] = sms_df["sms_received"].map({0: "No SMS", 1: "SMS Sent"})
        fig5 = px.bar(sms_df, x="SMS", y="no_show",
                      color="SMS",
                      color_discrete_map={"No SMS": "#e05c5c", "SMS Sent": "#3b82d4"},
                      text=sms_df["no_show"].apply(lambda x: f"{x*100:.1f}%"),
                      labels={"no_show": "No-Show Rate"})
        fig5.update_traces(textposition="outside")
        fig5.update_layout(showlegend=False, height=380, yaxis_tickformat=".0%")
        st.plotly_chart(fig5, use_container_width=True)

    # No-show by weekday
    st.subheader("No-Show Rate by Appointment Weekday")
    day_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    wd = df.groupby("appt_weekday")["no_show"].mean().reset_index()
    wd["Day"] = wd["appt_weekday"].map(day_map)
    fig6 = px.line(wd, x="Day", y="no_show", markers=True,
                   labels={"no_show": "No-Show Rate"}, line_shape="spline")
    fig6.update_traces(line_color="#3b82d4", marker_color="#e05c5c", marker_size=8)
    fig6.update_layout(height=320, yaxis_tickformat=".0%")
    st.plotly_chart(fig6, use_container_width=True)

    # Feature correlation
    st.subheader("Feature Correlation with No-Show")
    corr = df[FEATURE_COLS + ["no_show"]].corr()["no_show"].drop("no_show").sort_values()
    fig7 = px.bar(x=corr.values, y=corr.index, orientation="h",
                  color=corr.values,
                  color_continuous_scale=["#e05c5c", "#f7f8fa", "#3b82d4"],
                  labels={"x": "Pearson Correlation", "y": "Feature"})
    fig7.update_layout(height=420, coloraxis_showscale=False)
    st.plotly_chart(fig7, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — Model Performance
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance Dashboard")
    arts = load_artifacts()

    # Model comparison table
    st.subheader("Model Comparison")
    summary = arts["results_summary"]
    rows = []
    for name, m in summary.items():
        rows.append({
            "Model": name,
            "CV ROC-AUC": round(m["cv_roc"], 4),
            "Test ROC-AUC": round(m["test_roc"], 4),
            "Test Accuracy": f"{m['test_acc']*100:.2f}%",
            "Best": "✅" if name == arts["best_model_name"] else ""
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

    # Best model details
    st.markdown(f"### Best Model: **{arts['best_model_name']}**")

    # Recompute predictions for charts
    _, df = load_processed()
    X, y = get_X_y(df)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_sc = arts["scaler"].transform(X_test)
    y_pred    = arts["best_model"].predict(X_test_sc)
    y_proba   = arts["best_model"].predict_proba(X_test_sc)[:, 1]

    col1, col2 = st.columns(2)

    # Confusion Matrix
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm,
                           labels=dict(x="Predicted", y="Actual"),
                           x=["Showed Up", "No-Show"],
                           y=["Showed Up", "No-Show"],
                           color_continuous_scale="Blues",
                           text_auto=True)
        fig_cm.update_layout(height=380)
        st.plotly_chart(fig_cm, use_container_width=True)

    # ROC Curve
    with col2:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc_val = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"ROC (AUC={roc_auc_val:.4f})",
                                     line=dict(color="#3b82d4", width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     name="Random Classifier",
                                     line=dict(color="#e5e7eb", dash="dash")))
        fig_roc.update_layout(xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate",
                              height=380, showlegend=True)
        st.plotly_chart(fig_roc, use_container_width=True)

    # Feature Importance
    model = arts["best_model"]
    if hasattr(model, "feature_importances_"):
        st.subheader("Feature Importance")
        fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
        fig_fi = px.bar(x=fi.values, y=fi.index, orientation="h",
                        color=fi.values,
                        color_continuous_scale=["#f7f8fa", "#3b82d4"],
                        labels={"x": "Importance", "y": "Feature"})
        fig_fi.update_layout(height=450, coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)

    # Prediction probability distribution
    st.subheader("Predicted Probability Distribution")
    prob_df = pd.DataFrame({"Probability": y_proba,
                            "Actual": y_test.values})
    prob_df["Outcome"] = prob_df["Actual"].map({0: "Showed Up", 1: "No-Show"})
    fig_pd = px.histogram(prob_df, x="Probability", color="Outcome", nbins=50,
                          barmode="overlay", opacity=0.75,
                          color_discrete_map={"Showed Up": "#3b82d4", "No-Show": "#e05c5c"})
    fig_pd.update_layout(height=350)
    st.plotly_chart(fig_pd, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — Predict No-Show
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict No-Show":
    st.title("🔮 Predict Patient No-Show")
    st.markdown("Fill in the appointment details below to get an instant no-show risk assessment.")

    arts = load_artifacts()

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Patient Info")
            age        = st.slider("Age", 0, 115, 35)
            gender     = st.selectbox("Gender", ["Female", "Male"])
            scholarship = st.selectbox("Scholarship (Bolsa Família)", ["No", "Yes"])
            hipertension = st.selectbox("Hypertension", ["No", "Yes"])
            diabetes    = st.selectbox("Diabetes", ["No", "Yes"])

        with col2:
            st.markdown("#### Medical Details")
            alcoholism  = st.selectbox("Alcoholism", ["No", "Yes"])
            handcap     = st.selectbox("Handicap", ["No", "Yes"])
            sms_received = st.selectbox("SMS Reminder Sent", ["No", "Yes"])
            comorbidity_count = int(hipertension == "Yes") + int(diabetes == "Yes") + \
                                int(alcoholism == "Yes") + int(handcap == "Yes")
            st.info(f"Comorbidity Count: **{comorbidity_count}**")

        with col3:
            st.markdown("#### Appointment Details")
            wait_days   = st.slider("Days between Scheduling & Appointment", 0, 180, 7)
            appt_weekday = st.selectbox("Appointment Weekday",
                                        ["Monday", "Tuesday", "Wednesday",
                                         "Thursday", "Friday", "Saturday"])
            appt_month  = st.slider("Appointment Month", 1, 12, 4)
            neighbourhood_freq = st.slider(
                "Neighbourhood Frequency (0=rare, 1=common)", 0.0, 1.0, 0.05, step=0.01)

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted:
        weekday_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2,
                       "Thursday": 3, "Friday": 4, "Saturday": 5}
        age_group_map = {"child": 0, "teen": 1, "young_adult": 2, "adult": 3, "senior": 4}
        if   age <= 12: ag = 0
        elif age <= 17: ag = 1
        elif age <= 35: ag = 2
        elif age <= 60: ag = 3
        else:           ag = 4

        input_dict = {
            "age":               age,
            "scholarship":       int(scholarship == "Yes"),
            "hipertension":      int(hipertension == "Yes"),
            "diabetes":          int(diabetes == "Yes"),
            "alcoholism":        int(alcoholism == "Yes"),
            "handcap":           int(handcap == "Yes"),
            "sms_received":      int(sms_received == "Yes"),
            "wait_days":         wait_days,
            "appt_weekday":      weekday_map[appt_weekday],
            "appt_month":        appt_month,
            "comorbidity_count": comorbidity_count,
            "gender_enc":        int(gender == "Female"),
            "age_group_enc":     ag,
            "neighbourhood_freq": neighbourhood_freq,
        }

        row   = pd.DataFrame([input_dict])[arts["feature_cols"]]
        row_sc = arts["scaler"].transform(row)
        pred   = int(arts["best_model"].predict(row_sc)[0])
        prob   = float(arts["best_model"].predict_proba(row_sc)[0][1])

        st.markdown("---")
        col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
        with col_r2:
            if pred == 1:
                st.error(f"### ⚠️  High No-Show Risk")
                st.markdown(f"**No-show probability: {prob*100:.1f}%**")
            else:
                st.success(f"### ✅  Low No-Show Risk")
                st.markdown(f"**No-show probability: {prob*100:.1f}%**")

            # Gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#e05c5c" if pred else "#3b82d4"},
                    "steps": [
                        {"range": [0,  40], "color": "#d1fae5"},
                        {"range": [40, 65], "color": "#fef3c7"},
                        {"range": [65, 100], "color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line": {"color": "#1f2328", "width": 3},
                        "thickness": 0.8,
                        "value": prob * 100,
                    },
                },
                title={"text": "No-Show Probability"},
            ))
            fig_g.update_layout(height=320)
            st.plotly_chart(fig_g, use_container_width=True)

            # Risk factor breakdown
            st.markdown("#### Key Risk Factors")
            risk_factors = []
            if wait_days > 30: risk_factors.append(f"⏳ Long wait time ({wait_days} days)")
            if sms_received == "No": risk_factors.append("📵 No SMS reminder sent")
            if comorbidity_count == 0: risk_factors.append("🏥 No chronic conditions (less motivated)")
            if scholarship == "Yes": risk_factors.append("💰 Social assistance recipient")
            if appt_weekday in ["Friday", "Saturday"]: risk_factors.append("📅 End-of-week appointment")
            if age <= 12 or (18 <= age <= 30): risk_factors.append("👤 Age group with higher no-show tendency")

            if risk_factors:
                for f in risk_factors:
                    st.markdown(f"- {f}")
            else:
                st.markdown("- No major risk factors identified.")
