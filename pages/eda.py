import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

C = dict(
    primary="#00d4ff", success="#10b981", warning="#f59e0b",
    danger="#ef4444", purple="#7c3aed", border="#2a2f3e",
    text="#e2e8f0", muted="#8892a4", bg="#1c2333",
)
LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=C["text"], family="Inter,sans-serif", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor=C["border"], showgrid=True, color=C["muted"]),
    yaxis=dict(gridcolor=C["border"], showgrid=True, color=C["muted"]),
)


def _kpi(label, value, sub=""):
    return f"""<div class="kpi-card">
        <p class="kpi-label">{label}</p>
        <p class="kpi-value">{value}</p>
        {"<p class='kpi-sub'>" + sub + "</p>" if sub else ""}
    </div>"""


def render_eda(df: pd.DataFrame, metrics: dict):
    # ── KPI row ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(_kpi("Model",     metrics["model_name"],           "algorithm"),        unsafe_allow_html=True)
    c2.markdown(_kpi("Accuracy",  f"{metrics['accuracy']*100:.1f}%", "test set"),       unsafe_allow_html=True)
    c3.markdown(_kpi("AUC-ROC",   f"{metrics['auc']:.3f}",           "discrimination"), unsafe_allow_html=True)
    c4.markdown(_kpi("F1 Score",  f"{metrics['f1']:.3f}",            "harmonic mean"),  unsafe_allow_html=True)
    c5.markdown(_kpi("Recall",    f"{metrics['recall']*100:.1f}%",   "churn caught"),   unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROC + Confusion Matrix ────────────────────────────────────────────────
    st.markdown('<div class="section-title">Model Evaluation</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=metrics["fpr"], y=metrics["tpr"],
            mode="lines", name=f"AUC = {metrics['auc']:.3f}",
            line=dict(color=C["primary"], width=2.5),
            fill="tozeroy", fillcolor="rgba(0,212,255,0.07)",
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines",
            line=dict(color=C["muted"], dash="dash", width=1),
            showlegend=False,
        ))
        fig_roc.update_layout(**LAYOUT, title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(x=0.6, y=0.1, bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        cm = metrics["cm"]
        fig_cm = go.Figure(go.Heatmap(
            z=[[cm[1][1], cm[1][0]], [cm[0][1], cm[0][0]]],
            x=["Predicted: Churn", "Predicted: No Churn"],
            y=["Actual: Churn", "Actual: No Churn"],
            colorscale=[[0,"#161b27"],[1,"#00d4ff"]],
            text=[[str(cm[1][1]), str(cm[1][0])],[str(cm[0][1]), str(cm[0][0])]],
            texttemplate="%{text}", textfont=dict(size=18, color="#e2e8f0"),
            showscale=False,
        ))
        fig_cm.update_layout(**LAYOUT, title="Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── EDA section ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    # Restore original categorical columns for EDA from dummies
    # We'll work with what's available in df
    col3, col4 = st.columns(2)

    with col3:
        churn_counts = df["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        churn_counts["Label"] = churn_counts["Churn"].map({1: "Churned", 0: "Retained"})
        fig_pie = go.Figure(go.Pie(
            labels=churn_counts["Label"], values=churn_counts["Count"],
            marker=dict(colors=[C["danger"], C["success"]],
                        line=dict(color="#161b27", width=2)),
            hole=0.45,
            textinfo="percent+label",
            textfont=dict(color=C["text"], size=13),
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=C["text"]),
            margin=dict(l=10,r=10,t=40,b=10),
            title="Churn distribution",
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col4:
        fig_ten = go.Figure()
        for label, color, name in [(0, C["success"], "Retained"), (1, C["danger"], "Churned")]:
            subset = df[df["Churn"] == label]["tenure"]
            fig_ten.add_trace(go.Histogram(
                x=subset, name=name, opacity=0.75,
                marker_color=color, nbinsx=30,
            ))
        fig_ten.update_layout(**LAYOUT, barmode="overlay",
            title="Tenure distribution by churn",
            xaxis_title="Tenure (months)",
            yaxis_title="Count",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_ten, use_container_width=True)

    # Monthly charges vs tenure scatter
    st.markdown('<div class="section-title">Monthly Charges vs Tenure</div>', unsafe_allow_html=True)
    sample = df.sample(min(len(df), 600), random_state=42)
    fig_sc = go.Figure()
    for label, color, name in [(0, C["success"], "Retained"), (1, C["danger"], "Churned")]:
        sub = sample[sample["Churn"] == label]
        fig_sc.add_trace(go.Scatter(
            x=sub["tenure"], y=sub["MonthlyCharges"],
            mode="markers", name=name,
            marker=dict(color=color, size=5, opacity=0.65),
        ))
    fig_sc.update_layout(**LAYOUT,
        title="Do high charges + low tenure = more churn?",
        xaxis_title="Tenure (months)",
        yaxis_title="Monthly Charges ($)",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_sc, use_container_width=True)
