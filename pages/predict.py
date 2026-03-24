import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

C = dict(
    primary="#00d4ff", success="#10b981", warning="#f59e0b",
    danger="#ef4444", border="#2a2f3e", text="#e2e8f0", muted="#8892a4",
)
LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=C["text"], family="Inter,sans-serif", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor=C["border"], showgrid=True, color=C["muted"]),
    yaxis=dict(gridcolor=C["border"], showgrid=True, color=C["muted"]),
)

SAMPLE_CSV = """customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges
7590-VHVEG,Female,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85
5575-GNVDE,Male,0,No,No,34,Yes,No,DSL,Yes,No,Yes,No,No,No,One year,No,Mailed check,56.95,1889.5
3668-QPYBK,Male,0,No,No,2,Yes,No,DSL,Yes,Yes,No,No,No,No,Month-to-month,Yes,Mailed check,53.85,108.15
7795-CFOCW,Male,0,No,No,45,No,No phone service,DSL,Yes,No,Yes,Yes,No,No,One year,No,Bank transfer,42.30,1840.75
9237-HQITU,Female,0,No,No,2,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,70.70,151.65"""


def _risk_badge(prob: float) -> str:
    if prob >= 0.7:
        return f'<span class="risk-high">HIGH {prob*100:.0f}%</span>'
    elif prob >= 0.4:
        return f'<span class="risk-medium">MEDIUM {prob*100:.0f}%</span>'
    else:
        return f'<span class="risk-low">LOW {prob*100:.0f}%</span>'


def _preprocess_upload(df_up: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    df = df_up.copy()
    df.drop(columns=["customerID"], inplace=True, errors="ignore")
    df.drop(columns=["Churn"],      inplace=True, errors="ignore")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df["tenure"]         = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0)

    df["AvgMonthlySpend"]    = df["TotalCharges"] / (df["tenure"] + 1)
    df["IsNewCustomer"]      = (df["tenure"] <= 3).astype(int)
    df["IsLongTermCustomer"] = (df["tenure"] >= 48).astype(int)

    df = pd.get_dummies(df, drop_first=True)

    # Align columns with training features
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df


def render_predict(model, scaler, feature_cols: list):
    st.markdown('<div class="section-title">Upload Customer Data</div>', unsafe_allow_html=True)

    st.markdown(
        "<p style='color:#8892a4;font-size:0.88rem;margin-bottom:1rem;'>"
        "Upload a CSV with your customer data. The model will predict churn probability "
        "for each customer and flag high-risk accounts."
        "</p>",
        unsafe_allow_html=True
    )

    # ── Sample CSV download ───────────────────────────────────────────────────
    col_dl, col_info = st.columns([1, 3])
    with col_dl:
        st.download_button(
            label="Download sample CSV",
            data=SAMPLE_CSV,
            file_name="sample_customers.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_info:
        st.markdown(
            "<p style='color:#8892a4;font-size:0.82rem;padding-top:0.6rem;'>"
            "Not sure about the format? Download the sample CSV above, fill it with your data, then upload below."
            "</p>",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── File uploader ─────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Drop your CSV here",
        type=["csv"],
        help="Must contain columns: tenure, MonthlyCharges, TotalCharges, Contract, InternetService, etc.",
    )

    if uploaded is None:
        st.markdown(
            '<div class="insight-box">'
            '<strong style="color:#00d4ff">How it works</strong><br>'
            '1. Download the sample CSV to see the required format<br>'
            '2. Upload your customer CSV — must match the Telco dataset column names<br>'
            '3. The model returns a churn probability (0–100%) for every customer<br>'
            '4. Download the results CSV with all predictions attached'
            '</div>',
            unsafe_allow_html=True
        )
        return

    # ── Run predictions ───────────────────────────────────────────────────────
    try:
        df_up = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    if df_up.empty:
        st.warning("The uploaded file is empty.")
        return

    original_ids = df_up["customerID"].values if "customerID" in df_up.columns else [f"Customer_{i+1}" for i in range(len(df_up))]

    try:
        X_proc = _preprocess_upload(df_up, feature_cols)
        X_sc   = scaler.transform(X_proc)
        proba  = model.predict_proba(X_sc)[:, 1]
        preds  = (proba >= 0.5).astype(int)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Make sure your CSV has the same column names as the sample file.")
        return

    # ── Results ───────────────────────────────────────────────────────────────
    results = pd.DataFrame({
        "Customer ID":         original_ids,
        "Churn Probability":   (proba * 100).round(1),
        "Prediction":          ["Will Churn" if p == 1 else "Will Stay" for p in preds],
        "Risk Level":          ["High" if p >= 0.7 else "Medium" if p >= 0.4 else "Low" for p in proba],
    })

    # Summary KPIs
    n_total  = len(results)
    n_churn  = int(preds.sum())
    n_high   = int((proba >= 0.7).sum())
    avg_prob = float(proba.mean() * 100)

    st.markdown('<div class="section-title">Prediction Summary</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f'<div class="kpi-card"><p class="kpi-label">Total Customers</p><p class="kpi-value">{n_total}</p></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi-card"><p class="kpi-label">Predicted Churn</p><p class="kpi-value" style="color:#ef4444">{n_churn}</p><p class="kpi-sub" style="color:#ef4444">{n_churn/n_total*100:.0f}% of total</p></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi-card"><p class="kpi-label">High Risk</p><p class="kpi-value" style="color:#f59e0b">{n_high}</p><p class="kpi-sub" style="color:#f59e0b">prob ≥ 70%</p></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi-card"><p class="kpi-label">Avg Churn Prob</p><p class="kpi-value">{avg_prob:.1f}%</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Probability distribution chart
    st.markdown('<div class="section-title">Churn Probability Distribution</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=proba * 100, nbinsx=20,
        marker_color=C["primary"], opacity=0.8,
        name="Customers",
    ))
    fig.add_vline(x=50, line_dash="dash", line_color=C["danger"],
                  annotation_text="Decision threshold (50%)",
                  annotation_font_color=C["danger"])
    fig.update_layout(**LAYOUT,
        title="Distribution of churn probabilities",
        xaxis_title="Churn probability (%)",
        yaxis_title="Number of customers",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Results table ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Customer Predictions</div>', unsafe_allow_html=True)

    # Sort by probability descending
    results_sorted = results.sort_values("Churn Probability", ascending=False).reset_index(drop=True)
    results_sorted.index += 1
    st.dataframe(results_sorted, use_container_width=True, height=400)

    # ── Download button ───────────────────────────────────────────────────────
    csv_out = results_sorted.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download predictions CSV",
        data=csv_out,
        file_name="churn_predictions.csv",
        mime="text/csv",
        use_container_width=False,
    )
