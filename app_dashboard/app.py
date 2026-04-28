from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

# ── Chemins ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "src" / "models"
FIGURES    = ROOT / "reports" / "figures"

# ── Config page ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Marketing ROI Dashboard",
    page_icon="📊",
    layout="wide",
)

# ── Chargement des ressources (mis en cache) ──────────────────────────────────
@st.cache_resource
def load_artifacts():
    model        = joblib.load(MODELS_DIR / "marketing_model.joblib")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")

    df_raw      = pd.read_csv(ROOT / "data" / "Dummy Data HSS.csv").dropna(subset=["Sales"])
    x_background = preprocessor.transform(df_raw.drop("Sales", axis=1).sample(200, random_state=42))
    explainer   = shap.Explainer(model, x_background)

    return model, preprocessor, explainer

model, preprocessor, explainer = load_artifacts()

# ── Référence (valeurs moyennes de l'EDA) ────────────────────────────────────
REF = {"TV": 54.07, "Radio": 18.16, "Social Media": 3.33, "Influencer": "Mega"}

def predict_sales(tv, radio, social, influencer):
    df = pd.DataFrame([{"TV": tv, "Radio": radio,
                        "Social Media": social, "Influencer": influencer}])
    x  = preprocessor.transform(df)
    return float(model.predict(x)[0])

def compute_roi(sales, tv, radio, social):
    costs = tv + radio + social
    return (sales - costs) / costs if costs > 0 else 0.0

# ── Données de référence ──────────────────────────────────────────────────────
ref_sales = predict_sales(REF["TV"], REF["Radio"], REF["Social Media"], REF["Influencer"])
ref_roi   = compute_roi(ref_sales, REF["TV"], REF["Radio"], REF["Social Media"])

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("Paramètres budget")
    st.markdown("---")

    tv     = st.slider("TV",           min_value=10.0,  max_value=100.0,  value=54.07, step=0.5)
    radio  = st.slider("Radio",        min_value=0.0,   max_value=48.87,  value=18.16, step=0.5)
    social = st.slider("Social Media", min_value=0.0,   max_value=13.98,  value=3.33,  step=0.1)
    influencer = st.selectbox("Influenceur", ["Mega", "Macro", "Micro", "Nano"], index=0)

    st.markdown("---")
    st.caption("Valeurs de référence : budgets moyens observés sur le dataset.")

# ═════════════════════════════════════════════════════════════════════════════
# CALCULS SCENARIO ACTUEL
# ═════════════════════════════════════════════════════════════════════════════
cur_sales = predict_sales(tv, radio, social, influencer)
cur_roi   = compute_roi(cur_sales, tv, radio, social)

delta_sales = cur_sales - ref_sales
delta_roi   = cur_roi   - ref_roi

# ═════════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.title("Marketing ROI Dashboard")
st.markdown("Simulez l'impact de votre budget marketing sur les ventes et le ROI en temps réel.")
st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# MÉTRIQUES PRINCIPALES
# ═════════════════════════════════════════════════════════════════════════════
col1, col2, col3, col4 = st.columns(4)
col1.metric("Ventes prédites",  f"{cur_sales:.1f} k€", f"{delta_sales:+.1f} vs référence")
col2.metric("ROI prédit",       f"{cur_roi:.2%}",       f"{delta_roi:+.2%} vs référence")
col3.metric("Budget total",     f"{tv + radio + social:.1f} k€")
col4.metric("Modèle utilisé",  "XGBoost")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# GRAPHIQUE : SCÉNARIO ACTUEL vs RÉFÉRENCE
# ═════════════════════════════════════════════════════════════════════════════
left, right = st.columns(2)

with left:
    st.subheader("Scénario actuel vs Référence")

    bar_color = "#4C72B0" if cur_sales >= ref_sales else "#E63946"
    fig_bar = go.Figure(data=[
        go.Bar(
            x=["Référence (budget moyen)", "Scénario actuel"],
            y=[ref_sales, cur_sales],
            marker_color=["#ADB5BD", bar_color],
            text=[f"{ref_sales:.1f} k€", f"{cur_sales:.1f} k€"],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Ventes : %{y:.1f} k€<extra></extra>",
        )
    ])
    fig_bar.update_layout(
        yaxis_title="Ventes prédites (k€)",
        yaxis_range=[0, max(ref_sales, cur_sales) * 1.25],
        plot_bgcolor="white",
        margin=dict(t=20, b=10),
        height=350,
    )
    fig_bar.update_xaxes(showgrid=False)
    fig_bar.update_yaxes(showgrid=True, gridcolor="#F0F0F0")
    st.plotly_chart(fig_bar, use_container_width=True)

with right:
    st.subheader("Répartition du budget")

    fig_pie = go.Figure(data=[go.Pie(
        labels=["TV", "Radio", "Social Media"],
        values=[tv, radio, social],
        hole=0.42,
        marker_colors=["#4C72B0", "#55A868", "#C44E52"],
        textinfo="percent",
        textposition="inside",
        insidetextorientation="radial",
        hovertemplate="<b>%{label}</b><br>Budget : %{value:.1f} k€<br>Part : %{percent}<extra></extra>",
        pull=[0.03, 0.03, 0.03],
    )])
    fig_pie.update_layout(
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=0.5, xanchor="left"),
        margin=dict(t=20, b=20, l=20, r=120),
        height=350,
        annotations=[dict(
            text=f"<b>{tv + radio + social:.1f} k€</b>",
            x=0.5, y=0.5, font_size=15, showarrow=False
        )],
    )
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# ANALYSE SHAP
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("Analyse de la décision (SHAP)")
st.markdown("Impact de chaque canal marketing sur la prédiction actuelle.")

with st.spinner("Calcul SHAP en cours..."):
    df_input = pd.DataFrame([{"TV": tv, "Radio": radio,
                               "Social Media": social, "Influencer": influencer}])
    x_input  = preprocessor.transform(df_input)

    feature_names = ["TV", "Radio", "Social Media",
                     "Influencer_Macro", "Influencer_Mega",
                     "Influencer_Micro", "Influencer_Nano"]

    shap_values = explainer(x_input).values[0]

    sorted_idx   = np.argsort(np.abs(shap_values))
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_vals  = [shap_values[i]   for i in sorted_idx]
    bar_colors   = ["#E63946" if v > 0 else "#4C72B0" for v in sorted_vals]

    fig_shap = go.Figure(go.Bar(
        x=sorted_vals,
        y=sorted_names,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:+.3f}" for v in sorted_vals],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Contribution SHAP : %{x:+.4f}<extra></extra>",
    ))
    fig_shap.add_vline(x=0, line_color="black", line_width=1)
    fig_shap.update_layout(
        xaxis_title="Contribution SHAP (impact sur les ventes)",
        plot_bgcolor="white",
        margin=dict(t=10, b=10),
        height=320,
    )
    fig_shap.update_xaxes(showgrid=True, gridcolor="#F0F0F0", zeroline=False)
    fig_shap.update_yaxes(showgrid=False)
    st.plotly_chart(fig_shap, use_container_width=True)

st.caption("Rouge = augmente les ventes  |  Bleu = réduit les ventes")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# TABLEAU COMPARATIF DES MODÈLES
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("Comparatif des modèles entraînés")
st.markdown("Justification du choix du modèle de production.")

model_scores = pd.DataFrame([
    {"Modèle": "XGBoost ✅",          "CV R² moyen": 0.9954, "Test R²": 0.9987, "Test MAE": 2.64, "Test RMSE": 3.35},
    {"Modèle": "Random Forest",        "CV R² moyen": 0.9958, "Test R²": 0.9983, "Test MAE": 2.73, "Test RMSE": 3.82},
    {"Modèle": "Linear Regression",    "CV R² moyen": 0.9950, "Test R²": 0.9960, "Test MAE": 2.59, "Test RMSE": 5.88},
    {"Modèle": "Deep Learning (MLP)",  "CV R² moyen": 0.9942, "Test R²": 0.9951, "Test MAE": 3.36, "Test RMSE": 6.49},
])

st.dataframe(
    model_scores.style
        .highlight_max(subset=["CV R² moyen", "Test R²"], color="#d4edda")
        .highlight_min(subset=["Test MAE", "Test RMSE"],  color="#d4edda")
        .format({"CV R² moyen": "{:.4f}", "Test R²": "{:.4f}",
                 "Test MAE": "{:.2f}", "Test RMSE": "{:.2f}"}),
    use_container_width=True,
    hide_index=True,
)
st.caption("✅ = modèle sélectionné pour la production (meilleur Test R² et RMSE)")
