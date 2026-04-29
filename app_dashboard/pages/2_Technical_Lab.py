import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

ROOT    = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "reports" / "figures"
API_URL = "http://localhost:8000"

sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Technical Lab",
    layout="wide",
)

st.title("Monitoring Technique & Performance des Modèles")
st.markdown("Vue réservée à l'analyse approfondie des modèles, de leur explicabilité et de l'état de l'infrastructure.")
st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# Chargement des artefacts (mis en cache pour ne charger qu'une fois)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model_artifacts():
    model        = joblib.load(ROOT / "src" / "models" / "marketing_model.joblib")
    preprocessor = joblib.load(ROOT / "src" / "models" / "preprocessor.joblib")
    return model, preprocessor


@st.cache_data
def load_test_data():
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(ROOT / "data" / "Dummy Data HSS.csv")
    df = df.dropna(subset=["Sales"])
    _, df_test = train_test_split(df, test_size=0.2, random_state=42)
    return df_test


@st.cache_data
def get_preprocessed_test():
    from sklearn.model_selection import train_test_split
    from src.preprocessing.preprocessing import get_preprocessed_data
    _, x_test, _, y_test, _ = get_preprocessed_data()
    return x_test, y_test


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Comparaison des modèles
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("1. Comparaison des modèles")

model_scores = pd.DataFrame([
    {"Modèle": "XGBoost",            "CV R² moyen": 0.9954, "Test R²": 0.9987, "MAE": 2.64, "RMSE": 3.35},
    {"Modèle": "Random Forest",      "CV R² moyen": 0.9958, "Test R²": 0.9983, "MAE": 2.73, "RMSE": 3.82},
    {"Modèle": "Linear Regression",  "CV R² moyen": 0.9950, "Test R²": 0.9960, "MAE": 2.59, "RMSE": 5.88},
    {"Modèle": "Deep Learning (MLP)","CV R² moyen": 0.9942, "Test R²": 0.9951, "MAE": 3.36, "RMSE": 6.49},
])

st.dataframe(
    model_scores.style
        .highlight_max(subset=["CV R² moyen", "Test R²"], color="#d4edda")
        .highlight_min(subset=["MAE", "RMSE"], color="#d4edda")
        .format({
            "CV R² moyen": "{:.4f}",
            "Test R²":     "{:.4f}",
            "MAE":         "{:.2f}",
            "RMSE":        "{:.2f}",
        }),
    use_container_width=True,
    hide_index=True,
)

st.markdown(
    """
    **Modèle retenu : XGBoost**

    XGBoost présente le meilleur R² sur le jeu de test (0.9987) et le RMSE le plus faible (3.35 M€),
    ce qui signifie que ses prédictions s'écartent en moyenne de seulement **3,35 M€** des ventes réelles.
    La validation croisée 5-fold confirme la stabilité du modèle (écart-type R² < 0.002).

    Le MLP (Deep Learning) obtient les scores les plus faibles ici, ce qui s'explique par la nature
    tabulaire et relativement simple des données — les modèles à base d'arbres restent supérieurs
    dans ce contexte.
    """
)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Explicabilité globale (SHAP)
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("2. Explicabilité globale (SHAP)")

FEATURE_NAMES = ["TV", "Radio", "Social Media",
                 "Influencer_Macro", "Influencer_Mega",
                 "Influencer_Micro", "Influencer_Nano"]


@st.cache_data
def compute_shap_figure():
    import shap
    model, _ = load_model_artifacts()
    x_test, _ = get_preprocessed_test()

    explainer   = shap.Explainer(model, x_test)
    shap_values = explainer(x_test).values

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, x_test, feature_names=FEATURE_NAMES, show=False)
    fig = plt.gcf()
    return fig


try:
    shap_fig = compute_shap_figure()
    st.pyplot(shap_fig, use_container_width=True)
    plt.close("all")
except Exception as e:
    shap_path = FIGURES / "shap_summary.png"
    if shap_path.exists():
        from PIL import Image
        st.image(Image.open(shap_path), use_container_width=True)
        st.caption(f"Image pré-générée affichée (calcul à la volée indisponible : {e})")
    else:
        st.warning(f"Graphique SHAP indisponible : {e}")

st.markdown(
    """
    **Lecture du graphique**

    Ce graphique représente la contribution moyenne (en valeur absolue) de chaque canal
    sur l'ensemble du jeu de test.

    - La **TV** est de loin le moteur de croissance prédominant : une variation de budget TV
      entraîne les changements de ventes les plus importants.
    - La **Radio** occupe la deuxième place avec une influence modérée mais cohérente.
    - **Social Media** et les **types d'influenceurs** ont un impact marginal sur la prédiction,
      ce qui est cohérent avec la corrélation de 0.53 observée en EDA.

    Ces résultats orientent les recommandations budgétaires : concentrer les efforts sur la TV
    maximise le ROI, tandis qu'une sur-allocation Social Media présente un rendement décroissant.
    """
)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Analyse des résidus
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("3. Analyse des résidus")


@st.cache_data
def compute_residuals_figure():
    model, _ = load_model_artifacts()
    x_test, y_test = get_preprocessed_test()
    y_pred    = model.predict(x_test)
    residuals = y_test.values - y_pred

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residuals, bins=50, edgecolor="white", color="#4C72B0")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Zéro biais")
    ax.set_xlabel("Résidu (Ventes réelles − Ventes prédites)")
    ax.set_ylabel("Fréquence")
    ax.set_title("Distribution des résidus")
    ax.legend()
    fig.tight_layout()
    return fig, float(residuals.mean()), float(residuals.std())


try:
    res_fig, res_mean, res_std = compute_residuals_figure()
    st.pyplot(res_fig, use_container_width=True)
    plt.close("all")
except Exception as e:
    residuals_path = FIGURES / "residuals_distribution.png"
    if residuals_path.exists():
        from PIL import Image
        st.image(Image.open(residuals_path), use_container_width=True)
        st.caption(f"Image pré-générée affichée (calcul à la volée indisponible : {e})")
        res_mean, res_std = 0.007, 3.35
    else:
        st.warning(f"Graphique des résidus indisponible : {e}")
        res_mean, res_std = None, None

st.markdown(
    """
    **Lecture du graphique**

    La distribution des résidus (ventes réelles − ventes prédites) est centrée sur zéro
    (résidu moyen ≈ 0 M€), ce qui indique que **le modèle ne présente pas de biais systématique** :
    il ne surestime ni ne sous-estime les ventes de façon structurelle.

    La forme en cloche symétrique confirme que les erreurs sont aléatoires et non corrélées
    à une variable omise. L'écart-type des résidus correspond au RMSE calculé
    sur le jeu de test — les deux métriques sont cohérentes.
    """
)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Simulation de sensibilité Social Media
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("4. Simulation de sensibilité — Budget Social Media")

st.markdown(
    """
    Cette section simule l'impact d'une variation du budget **Social Media** sur le ROI
    et les ventes prédites, toutes choses égales par ailleurs.
    Le modèle XGBoost est appliqué sur le jeu de test avec le budget Social Media modifié.
    """
)

increase_pct = st.slider(
    "Variation du budget Social Media",
    min_value=-50,
    max_value=50,
    value=10,
    step=5,
    format="%d%%",
)

if st.button("Lancer la simulation", type="primary"):
    try:
        model, preprocessor = load_model_artifacts()
        df_test_raw         = load_test_data()

        df_base     = df_test_raw.copy()
        df_scenario = df_test_raw.copy()
        df_scenario["Social Media"] = df_scenario["Social Media"] * (1 + increase_pct / 100)

        x_base     = preprocessor.transform(df_base.drop("Sales", axis=1))
        x_scenario = preprocessor.transform(df_scenario.drop("Sales", axis=1))

        pred_base     = model.predict(x_base)
        pred_scenario = model.predict(x_scenario)

        costs_base     = df_base[["TV", "Radio", "Social Media"]].sum(axis=1)
        costs_scenario = df_scenario[["TV", "Radio", "Social Media"]].sum(axis=1)
        roi_base       = ((pred_base - costs_base) / costs_base).mean()
        roi_scenario   = ((pred_scenario - costs_scenario) / costs_scenario).mean()
        roi_delta      = roi_scenario - roi_base
        roi_delta_pct  = roi_delta / abs(roi_base) * 100
        sales_gain     = (pred_scenario - pred_base).mean()

        sign = "+" if increase_pct >= 0 else ""
        st.markdown(f"#### Résultats — Scénario {sign}{increase_pct}% Social Media")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROI moyen — Base",    f"{roi_base:.2%}")
        c2.metric("ROI moyen — Scénario",f"{roi_scenario:.2%}",
                  delta=f"{roi_delta:+.2%}")
        c3.metric("Δ ROI",               f"{roi_delta_pct:+.2f}%")
        c4.metric("Gain ventes moyen",   f"{sales_gain:+.2f} M€")

        st.markdown(
            f"""
            **Interprétation**

            Une {"hausse" if increase_pct >= 0 else "baisse"} de **{abs(increase_pct)}%** du budget Social Media
            entraîne un gain de ventes moyen de **{sales_gain:+.2f} M€** par campagne et fait évoluer
            le ROI de **{roi_base:.2%}** à **{roi_scenario:.2%}** ({roi_delta_pct:+.2f}%).

            Compte tenu du faible poids de Social Media dans les valeurs SHAP (section 2),
            {"cet investissement supplémentaire présente un rendement marginal limité." if increase_pct > 0 else "cette réduction budgétaire a un impact limité sur les ventes prédites."}
            """
        )

    except Exception as e:
        st.error(f"Erreur lors de la simulation : {e}")
else:
    st.info("Ajustez le curseur puis cliquez sur **Lancer la simulation** pour calculer l'impact.")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Statut de l'infrastructure
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("5. Statut de l'infrastructure")

col_api, col_model = st.columns(2)

with col_api:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        if resp.status_code == 200:
            st.success("API FastAPI — Active  |  `localhost:8000`")
            data = resp.json()
            st.json(data)
        else:
            st.error(f"API FastAPI — Erreur HTTP {resp.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("API FastAPI — Hors ligne")
        st.caption("Le dashboard fonctionne en mode fallback : inférence locale via `marketing_model.joblib`.")
    except requests.exceptions.Timeout:
        st.warning("API FastAPI — Timeout (> 3s)")

with col_model:
    model_file   = ROOT / "src" / "models" / "marketing_model.joblib"
    preproc_file = ROOT / "src" / "models" / "preprocessor.joblib"

    st.info("Modèle local")
    st.markdown(f"- `marketing_model.joblib` : {'présent' if model_file.exists() else 'manquant'}")
    st.markdown(f"- `preprocessor.joblib` : {'présent' if preproc_file.exists() else 'manquant'}")
    if model_file.exists():
        size_kb = model_file.stat().st_size / 1024
        st.caption(f"Taille du modèle : {size_kb:.1f} Ko")
