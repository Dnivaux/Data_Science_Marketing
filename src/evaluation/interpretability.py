from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.preprocessing import get_preprocessed_data, load_data, DATA_PATH

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
MODELS_DIR    = PROJECT_ROOT / "src" / "models"
FIGURES_DIR   = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = ["TV", "Radio", "Social Media", "Influencer_Macro",
                 "Influencer_Mega", "Influencer_Micro", "Influencer_Nano"]


def load_artifacts():
    model       = joblib.load(MODELS_DIR / "marketing_model.joblib")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    return model, preprocessor


def calculate_marketing_roi(df_input: pd.DataFrame, predictions: np.ndarray) -> pd.Series:
    costs = df_input[["TV", "Radio", "Social Media"]].sum(axis=1)
    roi   = (predictions - costs) / costs
    return roi


def shap_analysis(model, x_test_preprocessed: np.ndarray):
    explainer   = shap.Explainer(model, x_test_preprocessed)
    shap_values = explainer(x_test_preprocessed).values

    plt.figure()
    shap.summary_plot(
        shap_values,
        x_test_preprocessed,
        feature_names=FEATURE_NAMES,
        show=False,
    )
    path = FIGURES_DIR / "shap_summary.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"SHAP summary plot sauvegardé : {path}")
    return shap_values


def residuals_analysis(y_test: pd.Series, y_pred: np.ndarray):
    residuals = y_test.values - y_pred

    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=50, edgecolor="white", color="#4C72B0")
    plt.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Zéro biais")
    plt.xlabel("Résidu (Ventes réelles − Ventes prédites)")
    plt.ylabel("Fréquence")
    plt.title("Distribution des résidus")
    plt.legend()

    path = FIGURES_DIR / "residuals_distribution.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Distribution des résidus sauvegardée : {path}")

    print(f"  Résidu moyen  : {residuals.mean():.4f}  (proche de 0 = pas de biais)")
    print(f"  Écart-type    : {residuals.std():.4f}")
    return residuals


def simulate_social_media_increase(
    model,
    preprocessor,
    df_raw: pd.DataFrame,
    increase_pct: float = 0.10,
) -> dict:
    df_base     = df_raw.copy()
    df_scenario = df_raw.copy()
    df_scenario["Social Media"] = df_scenario["Social Media"] * (1 + increase_pct)

    x_base     = preprocessor.transform(df_base.drop("Sales", axis=1))
    x_scenario = preprocessor.transform(df_scenario.drop("Sales", axis=1))

    pred_base     = model.predict(x_base)
    pred_scenario = model.predict(x_scenario)

    roi_base     = calculate_marketing_roi(df_base, pred_base).mean()
    roi_scenario = calculate_marketing_roi(df_scenario, pred_scenario).mean()
    roi_delta    = roi_scenario - roi_base

    result = {
        "roi_base":         round(roi_base, 4),
        "roi_scenario":     round(roi_scenario, 4),
        "roi_delta":        round(roi_delta, 4),
        "roi_delta_pct":    round(roi_delta / abs(roi_base) * 100, 2),
        "sales_gain_mean":  round((pred_scenario - pred_base).mean(), 4),
    }

    print(f"\n--- Simulateur ROI : +{int(increase_pct*100)}% budget Social Media ---")
    print(f"  ROI moyen base      : {result['roi_base']:.4f}")
    print(f"  ROI moyen scénario  : {result['roi_scenario']:.4f}")
    print(f"  Delta ROI           : {result['roi_delta']:+.4f}  ({result['roi_delta_pct']:+.2f}%)")
    print(f"  Gain ventes moyen   : +{result['sales_gain_mean']:.2f}")
    return result


def run_evaluation():
    model, preprocessor = load_artifacts()

    x_train, x_test, y_train, y_test, _ = get_preprocessed_data()

    df_raw  = load_data(DATA_PATH)
    _, df_test_raw = __import__("sklearn.model_selection", fromlist=["train_test_split"]).train_test_split(
        df_raw, test_size=0.2, random_state=42
    )

    y_pred = model.predict(x_test)

    print("=== Analyse SHAP ===")
    shap_analysis(model, x_test)

    print("\n=== ROI Marketing (jeu de test) ===")
    roi = calculate_marketing_roi(df_test_raw, y_pred)
    print(f"  ROI moyen prédit : {roi.mean():.4f}")
    print(f"  ROI médian prédit: {roi.median():.4f}")

    print("\n=== Analyse des résidus ===")
    residuals_analysis(y_test, y_pred)

    print("\n=== Simulation sensibilité ===")
    simulate_social_media_increase(model, preprocessor, df_test_raw)


if __name__ == "__main__":
    run_evaluation()
