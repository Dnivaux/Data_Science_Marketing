from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Technical Lab",
    layout="wide",
)

ROOT    = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "reports" / "figures"
API_URL = "http://localhost:8000"

st.title("Monitoring Technique & Performance des Modèles")
st.markdown("Vue réservée à l'analyse approfondie des modèles, de leur explicabilité et de l'état de l'infrastructure.")
st.markdown("---")

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

    XGBoost présente le meilleur R² sur le jeu de test (0.9987) et le RMSE le plus faible (3.35 k€),
    ce qui signifie que ses prédictions s'écartent en moyenne de seulement **3 350 €** des ventes réelles.
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

shap_path = FIGURES / "shap_summary.png"
if shap_path.exists():
    st.image(Image.open(shap_path), use_container_width=True)
else:
    st.warning("Graphique SHAP introuvable. Lancez d'abord `src/evaluation/interpretability.py`.")

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

residuals_path = FIGURES / "residuals_distribution.png"
if residuals_path.exists():
    st.image(Image.open(residuals_path), use_container_width=True)
else:
    st.warning("Graphique des résidus introuvable. Lancez d'abord `src/evaluation/interpretability.py`.")

st.markdown(
    """
    **Lecture du graphique**

    La distribution des résidus (ventes réelles − ventes prédites) est centrée sur zéro
    (résidu moyen = 0.007 k€), ce qui indique que **le modèle ne présente pas de biais systématique** :
    il ne surestime ni ne sous-estime les ventes de façon structurelle.

    La forme en cloche symétrique confirme que les erreurs sont aléatoires et non corrélées
    à une variable omise. L'écart-type des résidus (3.35 k€) correspond au RMSE calculé
    sur le jeu de test — les deux métriques sont cohérentes.
    """
)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Statut de l'infrastructure
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("4. Statut de l'infrastructure")

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
    from pathlib import Path as _Path
    model_file = ROOT / "src" / "models" / "marketing_model.joblib"
    preproc_file = ROOT / "src" / "models" / "preprocessor.joblib"

    st.info("Modèle local")
    st.markdown(f"- `marketing_model.joblib` : {'présent' if model_file.exists() else 'manquant'}")
    st.markdown(f"- `preprocessor.joblib` : {'présent' if preproc_file.exists() else 'manquant'}")
    if model_file.exists():
        size_kb = model_file.stat().st_size / 1024
        st.caption(f"Taille du modèle : {size_kb:.1f} Ko")
