from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.preprocessing import get_preprocessed_data, get_preprocessing_pipeline, DATA_PATH, load_data

MODELS_DIR = Path(__file__).resolve().parent


def _sanity_check(model, preprocessor):
    """Vérifie que le modèle prédit des valeurs réalistes sur des budgets moyens."""
    checks = [
        {"TV": 50.0, "Radio": 20.0, "Social Media": 3.3,  "Influencer": "Mega",  "expected_min": 100},
        {"TV": 80.0, "Radio": 30.0, "Social Media": 5.0,  "Influencer": "Macro", "expected_min": 200},
        {"TV": 10.0, "Radio": 1.0,  "Social Media": 0.5,  "Influencer": "Nano",  "expected_min": 30},
    ]
    print("\n=== Vérification de sanité post-entraînement ===")
    all_ok = True
    for c in checks:
        expected_min = c.pop("expected_min")
        df  = pd.DataFrame([c])
        x   = preprocessor.transform(df)
        pred = float(model.predict(x)[0])
        ok  = pred > expected_min
        status = "OK" if ok else "ERREUR"
        print(f"  [{status}] TV={c['TV']}, Radio={c['Radio']} → prédiction = {pred:.2f} k€  (attendu > {expected_min})")
        if not ok:
            all_ok = False
    if not all_ok:
        raise ValueError("La vérification de sanité a échoué : prédictions incohérentes. Ne pas utiliser ce modèle.")
    print("  Toutes les vérifications sont passées.")
    return True


def train_model():
    # ── Données ───────────────────────────────────────────────────────────────
    x_train, x_test, y_train, y_test, preprocessor = get_preprocessed_data(
        save_preprocessor=False  # on sauvegarde manuellement après validation
    )

    # ── Entraînement des 4 modèles ────────────────────────────────────────────
    models = {
        "Linear Regression":   LinearRegression(),
        "Random Forest":       RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost":             XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        "Deep Learning (MLP)": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            early_stopping=True,
            random_state=42,
        ),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring="r2")
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        r2   = r2_score(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append({
            "Modèle":      name,
            "CV R² moyen": round(cv_scores.mean(), 4),
            "CV R² std":   round(cv_scores.std(), 4),
            "Test R²":     round(r2, 4),
            "Test MAE":    round(mae, 4),
            "Test RMSE":   round(rmse, 4),
        })
        trained_models[name] = (model, y_pred, r2)

    df_results = pd.DataFrame(results).sort_values("Test R²", ascending=False)
    print("\n=== Tableau comparatif des modèles ===")
    print(df_results.to_string(index=False))

    best_name               = df_results.iloc[0]["Modèle"]
    best_model, best_pred, best_r2 = trained_models[best_name]
    print(f"\nMeilleur modèle : {best_name} (Test R²: {best_r2:.4f})")

    # ── Vérification de sanité AVANT sauvegarde ───────────────────────────────
    _sanity_check(best_model, preprocessor)

    # ── Sauvegarde atomique des deux artefacts ────────────────────────────────
    # On sauvegarde preprocessor ET modèle ensemble pour garantir leur cohérence.
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    model_path        = MODELS_DIR / "marketing_model.joblib"

    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(best_model,   model_path)
    print(f"\nArtefacts sauvegardés (sklearn {__import__('sklearn').__version__}) :")
    print(f"  {preprocessor_path}")
    print(f"  {model_path}")

    # ── Analyse des erreurs ───────────────────────────────────────────────────
    residuals = y_test.values - best_pred
    top5_idx  = np.argsort(np.abs(residuals))[-5:][::-1]
    print(f"\n=== 5 plus grosses erreurs ({best_name}) ===")
    df_errors = pd.DataFrame({
        "Ventes réelles":  y_test.values[top5_idx],
        "Ventes prédites": best_pred[top5_idx].round(2),
        "Résidu":          residuals[top5_idx].round(2),
    })
    print(df_errors.to_string(index=False))

    return best_model, df_results


if __name__ == "__main__":
    train_model()
