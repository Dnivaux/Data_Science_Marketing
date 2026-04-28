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

from src.preprocessing.preprocessing import get_preprocessed_data

MODELS_DIR = Path(__file__).resolve().parent


def train_model():
    x_train, x_test, y_train, y_test, _ = get_preprocessed_data(save_preprocessor=True)

    models = {
        "Linear Regression":  LinearRegression(),
        "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost":            XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
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
            "Modèle":          name,
            "CV R² moyen":     round(cv_scores.mean(), 4),
            "CV R² std":       round(cv_scores.std(), 4),
            "Test R²":         round(r2, 4),
            "Test MAE":        round(mae, 4),
            "Test RMSE":       round(rmse, 4),
        })
        trained_models[name] = (model, y_pred, r2)

    df_results = pd.DataFrame(results).sort_values("Test R²", ascending=False)
    print("\n=== Tableau comparatif des modèles ===")
    print(df_results.to_string(index=False))

    best_name  = df_results.iloc[0]["Modèle"]
    best_model, best_pred, best_r2 = trained_models[best_name]
    print(f"\nMeilleur modèle : {best_name} (Test R²: {best_r2:.4f})")

    model_path = MODELS_DIR / "marketing_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"Modèle sauvegardé : {model_path}")

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
