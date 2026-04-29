import streamlit as st
import requests

API_URL = "http://localhost:8000"


def get_api_prediction(tv: float, radio: float, social: float, influencer: str) -> float | None:
    payload = {
        "tv": tv,
        "radio": radio,
        "social_media": social,
        "influencer": influencer,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        response.raise_for_status()
        data = response.json()
        if "prediction" not in data:
            st.error(f"Erreur API : {data.get('error', data)}")
            return None
        return data["prediction"]
    except requests.exceptions.ConnectionError:
        return None  # API non démarrée — l'appelant gère le fallback silencieusement
    except requests.exceptions.Timeout:
        st.warning("API — Timeout (5s), bascule sur le modèle local.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"Erreur API ({response.status_code}) : {e}")
        return None


def format_currency(value: float, decimals: int = 1) -> str:
    return f"{value:,.{decimals}f} k€".replace(",", " ")


def compute_roi(sales: float, tv: float, radio: float, social: float) -> float:
    costs = tv + radio + social
    return (sales - costs) / costs if costs > 0 else 0.0


def format_roi(roi: float) -> str:
    return f"{roi:.2%}"


def roi_delta_label(roi_current: float, roi_reference: float) -> str:
    delta = roi_current - roi_reference
    sign  = "+" if delta >= 0 else ""
    return f"{sign}{delta:.2%} vs référence"
