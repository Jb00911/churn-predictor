import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

C = dict(
    primary="#00d4ff", success="#10b981", warning="#f59e0b",
    danger="#ef4444", purple="#7c3aed", border="#2a2f3e",
    text="#e2e8f0", muted="#8892a4",
)
LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=C["text"], family="Inter,sans-serif", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor=C["border"], showgrid=True, color=C["muted"]),
    yaxis=dict(gridcolor=C["border"], showgrid=True, color=C["muted"]),
)

RECOMMENDATIONS = [
    ("Month-to-month contracts drive the most churn",
     "Offer 10–15% discount to customers on monthly plans who upgrade to annual. A small price cut is cheaper than replacing a churned customer."),
    ("New customers (tenure < 3 months) are highest risk",
     "Assign a dedicated onboarding specialist to customers in their first 90 days. A single check-in call reduces early churn significantly."),
    ("High monthly charges correlate with churn",
     "Review pricing tiers above $65/month. Consider a loyalty discount for long-term customers paying premium rates."),
    ("Customers without TechSupport churn more",
     "Bundle TechSupport into standard plans or offer a free trial month. The retention value outweighs the cost."),
    ("Electronic check users churn at higher rates",
     "Incentivise auto-pay via bank transfer or credit card with a small monthly credit. This also reduces payment failures."),
]


def render_insights(model, scaler, feature_cols: list, df: pd.DataFrame):

    # ── Feature Importance ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Top Churn Drivers</div>', unsafe_allow_html=True)

    has_native = hasattr(model, "feature_importances_")
    if has_native:
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0]) if hasattr(model, "coef_") else np.ones(len(feature_cols))

    feat_df = (
        pd.DataFrame({"Feature": feature_cols, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(20)
    )

    # Clean up feature names for display
    feat_df["Feature"] = (
        feat_df["Feature"]
        .str.replace("_", " ")
        .str.replace("Yes", "")
        .str.strip()
        .str.title()
    )

    fig_imp = go.Figure(go.Bar(
        x=feat_df["Importance"][::-1],
        y=feat_df["Feature"][::-1],
        orientation="h",
        marker=dict(
            color=feat_df["Importance"][::-1],
            colorscale=[[0, "#0f3460"], [1, C["primary"]]],
            showscale=False,
        ),
    ))
    fig_imp.update_layout(
        **LAYOUT,
        title="Top 20 features driving churn predictions",
        xaxis_title="Importance score",
        height=520,
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # ── Risk Segments ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Who Churns Most?</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        # tenure buckets
        df2 = df.copy()
        df2["Tenure Group"] = pd.cut(
            df2["tenure"],
            bins=[0, 3, 12, 24, 48, 999],
            labels=["0–3 mo", "4–12 mo", "13–24 mo", "25–48 mo", "48+ mo"]
        )
        seg = df2.groupby("Tenure Group", observed=True)["Churn"].mean().reset_index()
        seg.columns = ["Tenure Group", "Churn Rate"]
        seg["Churn Rate %"] = (seg["Churn Rate"] * 100).round(1)

        colors = [C["danger"] if r > 0.35 else C["warning"] if r > 0.2 else C["success"]
                  for r in seg["Churn Rate"]]
        fig_seg = go.Figure(go.Bar(
            x=seg["Tenure Group"].astype(str),
            y=seg["Churn Rate %"],
            marker_color=colors,
            text=seg["Churn Rate %"].map("{:.1f}%".format),
            textposition="outside",
            textfont=dict(color=C["text"]),
        ))
        layout_seg = {**LAYOUT}
        layout_seg["yaxis"] = dict(gridcolor=C["border"], showgrid=True, color=C["muted"], range=[0, 70], title="Churn rate (%)")
        fig_seg.update_layout(**layout_seg, title="Churn rate by tenure group")
        st.plotly_chart(fig_seg, use_container_width=True)

    with col2:
        # Monthly charge buckets
        df2["Charge Group"] = pd.cut(
            df2["MonthlyCharges"],
            bins=[0, 35, 55, 75, 9999],
            labels=["<$35", "$35–55", "$55–75", ">$75"]
        )
        chg = df2.groupby("Charge Group", observed=True)["Churn"].mean().reset_index()
        chg.columns = ["Charge Group", "Churn Rate"]
        chg["Churn Rate %"] = (chg["Churn Rate"] * 100).round(1)

        colors2 = [C["danger"] if r > 0.35 else C["warning"] if r > 0.2 else C["success"]
                   for r in chg["Churn Rate"]]
        fig_chg = go.Figure(go.Bar(
            x=chg["Charge Group"].astype(str),
            y=chg["Churn Rate %"],
            marker_color=colors2,
            text=chg["Churn Rate %"].map("{:.1f}%".format),
            textposition="outside",
            textfont=dict(color=C["text"]),
        ))
        layout_chg = {**LAYOUT}
        layout_chg["yaxis"] = dict(gridcolor=C["border"], showgrid=True, color=C["muted"], range=[0, 70], title="Churn rate (%)")
        fig_chg.update_layout(**layout_chg, title="Churn rate by monthly charge band")
        st.plotly_chart(fig_chg, use_container_width=True)

    # ── Business Recommendations ──────────────────────────────────────────────
    st.markdown('<div class="section-title">Business Recommendations</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8892a4;font-size:0.88rem;margin-bottom:1rem;'>"
        "Actionable retention strategies derived from model findings."
        "</p>",
        unsafe_allow_html=True
    )

    for i, (title, body) in enumerate(RECOMMENDATIONS, 1):
        st.markdown(
            f'<div class="insight-box">'
            f'<strong style="color:#00d4ff">{i}. {title}</strong><br>'
            f'<span style="color:#c9d1d9">{body}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
