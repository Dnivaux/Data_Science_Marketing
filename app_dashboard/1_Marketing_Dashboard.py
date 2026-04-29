import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

ROOT       = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "src" / "models"
FIGURES    = ROOT / "reports" / "figures"

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_api_prediction

# ── Config page ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Marketing Dashboard",
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
    """Tente l'API, bascule sur le modèle local si indisponible.
    Retourne (valeur_prédite, source) où source vaut 'API' ou 'local'."""
    api_result = get_api_prediction(tv, radio, social, influencer)
    if api_result is not None:
        return api_result, "API"
    df = pd.DataFrame([{"TV": tv, "Radio": radio,
                        "Social Media": social, "Influencer": influencer}])
    x  = preprocessor.transform(df)
    return float(model.predict(x)[0]), "local"

def compute_roi(sales, tv, radio, social):
    costs = tv + radio + social
    return (sales - costs) / costs if costs > 0 else 0.0

# ── Données de référence ──────────────────────────────────────────────────────
ref_sales, _ = predict_sales(REF["TV"], REF["Radio"], REF["Social Media"], REF["Influencer"])
ref_roi      = compute_roi(ref_sales, REF["TV"], REF["Radio"], REF["Social Media"])

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("Paramètres budget")
    st.caption("Les budgets sont exprimés en millions d'euros (M€).")
    st.markdown("---")

    tv     = st.slider("TV (M€)",           min_value=10.0,  max_value=100.0,  value=54.07, step=0.5)
    st.caption(f"Budget TV actuel : **{tv:.1f} M€** — plage observée : 10 à 100 M€")

    radio  = st.slider("Radio (M€)",        min_value=0.0,   max_value=48.87,  value=18.16, step=0.5)
    st.caption(f"Budget Radio actuel : **{radio:.1f} M€** — plage observée : 0 à 48.9 M€")

    social = st.slider("Social Media (M€)", min_value=0.0,   max_value=13.98,  value=3.33,  step=0.1)
    st.caption(f"Budget Social Media actuel : **{social:.1f} M€** — plage observée : 0 à 14 M€")

    st.markdown("---")
    influencer = st.selectbox("Type d'influenceur", ["Mega", "Macro", "Micro", "Nano"], index=0)
    st.caption("Mega : >1M abonnés · Macro : 100K–1M · Micro : 10K–100K · Nano : <10K")

    st.markdown("---")
    st.caption("Référence : budgets moyens observés sur le dataset (TV 54 M€, Radio 18 M€, Social 3.3 M€).")

# ═════════════════════════════════════════════════════════════════════════════
# CALCULS SCENARIO ACTUEL
# ═════════════════════════════════════════════════════════════════════════════
cur_sales, pred_source = predict_sales(tv, radio, social, influencer)
cur_roi                = compute_roi(cur_sales, tv, radio, social)

if pred_source == "API":
    st.sidebar.success("Inférence via **API FastAPI** `localhost:8000`")
else:
    st.sidebar.info("Inférence via **modèle local** (API hors ligne)")

delta_sales = cur_sales - ref_sales
delta_roi   = cur_roi   - ref_roi

# ═════════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.title("Marketing Dashboard")
st.markdown("Simulez l'impact de votre budget marketing sur les ventes et le ROI en temps réel.")
st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# MÉTRIQUES PRINCIPALES
# ═════════════════════════════════════════════════════════════════════════════
col1, col2, col3, col4 = st.columns(4)
col1.metric("Ventes prédites",  f"{cur_sales:.1f} M€", f"{delta_sales:+.1f} vs référence")
col2.metric("ROI prédit",       f"{cur_roi:.2%}",       f"{delta_roi:+.2%} vs référence")
col3.metric("Budget total",     f"{tv + radio + social:.1f} M€")
col4.metric("Modèle utilisé",  f"XGBoost ({pred_source})")

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
            text=[f"{ref_sales:.1f} M€", f"{cur_sales:.1f} M€"],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Ventes : %{y:.1f} M€<extra></extra>",
        )
    ])
    fig_bar.update_layout(
        yaxis_title="Ventes prédites (M€)",
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
        hovertemplate="<b>%{label}</b><br>Budget : %{value:.1f} M€<br>Part : %{percent}<extra></extra>",
        pull=[0.03, 0.03, 0.03],
    )])
    fig_pie.update_layout(
        showlegend=True,
        legend=dict(orientation="v", x=1.02, y=0.5, xanchor="left"),
        margin=dict(t=20, b=20, l=20, r=120),
        height=350,
        annotations=[dict(
            text=f"<b>{tv + radio + social:.1f} M€</b>",
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
