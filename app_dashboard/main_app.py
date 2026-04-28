import streamlit as st

st.set_page_config(
    page_title="Marketing AI — Accueil",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("IA & Optimisation du ROI Marketing")
st.markdown(
    "Système de prédiction et d'aide à la décision pour l'allocation des budgets marketing."
)
st.markdown("---")

# ── Résumé du projet ──────────────────────────────────────────────────────────
st.subheader("Contexte du projet")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown(
        """
        Ce projet analyse l'impact des canaux marketing **(TV, Radio, Social Media, Influenceurs)**
        sur les ventes afin d'optimiser l'allocation budgétaire.

        Le pipeline complet couvre :
        - **Étape 1 — Prétraitement** : nettoyage, imputation, normalisation et encodage des données
        - **Étape 2 — Entraînement** : comparaison de 4 modèles (Régression Linéaire, Random Forest, XGBoost, MLP)
        - **Étape 3 — Interprétabilité** : analyse SHAP, calcul du ROI et simulation de sensibilité
        - **Étape 4 — Dashboard** : interface interactive de simulation en temps réel
        """
    )

with col_right:
    st.info(
        """
        **Dataset**
        4 572 observations

        **Meilleur modèle**
        XGBoost — R² = 0.9987

        **ROI moyen prédit**
        +154 % sur le jeu de test
        """
    )

st.markdown("---")

# ── Navigation ────────────────────────────────────────────────────────────────
st.subheader("Navigation")
st.markdown("L'application est organisée en deux vues accessibles depuis le menu latéral.")

nav_left, nav_right = st.columns(2)

with nav_left:
    st.markdown("#### Business Cockpit")
    st.markdown(
        """
        Vue orientée **décision métier**.

        - Simulateur de budget en temps réel (sliders TV / Radio / Social Media / Influenceur)
        - Prédiction des ventes et calcul du ROI associé
        - Comparaison scénario actuel vs référence (budget moyen)
        - Répartition visuelle du budget par canal

        Destiné aux **équipes marketing et direction** pour piloter les allocations budgétaires.
        """
    )

with nav_right:
    st.markdown("#### Technical Lab")
    st.markdown(
        """
        Vue orientée **analyse des modèles**.

        - Analyse SHAP : contribution de chaque canal sur la prédiction courante
        - Tableau comparatif des 4 modèles (CV R², MAE, RMSE)
        - Distribution des résidus pour détecter les biais systématiques
        - Simulation de sensibilité (+10 % Social Media → impact ROI)

        Destiné aux **Data Scientists et équipes techniques** pour auditer et valider le système.
        """
    )

st.markdown("---")

# ── Architecture API ──────────────────────────────────────────────────────────
st.subheader("Architecture technique")

st.markdown(
    """
    Le système repose sur une architecture découplée en deux couches :

    | Couche | Technologie | Rôle |
    |---|---|---|
    | **Inférence** | FastAPI (`localhost:8000`) | Expose le modèle XGBoost via une API REST (`POST /predict`) |
    | **Interface** | Streamlit | Consomme l'API et affiche les résultats en temps réel |

    Cette séparation permet de **réutiliser l'API** indépendamment du dashboard
    (intégration CRM, scripts batch, autres interfaces).
    """
)

st.info(
    "Le dashboard peut fonctionner en mode autonome (modèle chargé localement) "
    "ou en mode connecté (inférence déléguée à l'API FastAPI). "
    "Si l'API est éteinte, le système bascule automatiquement sur le modèle local."
)
