from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.preprocessing import get_preprocessed_data

MODELS_DIR = Path(__file__).resolve().parent


def train_model():
    x_train, x_test, y_train, y_test, _ = get_preprocessed_data(save_preprocessor=True)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        results[name] = {"model": model, "r2": r2, "mse": mse}
        print(f"{name} — R²: {r2:.4f} | MSE: {mse:.4f}")

    best_name = max(results, key=lambda n: results[n]["r2"])
    best_model = results[best_name]["model"]
    print(f"\nMeilleur modèle : {best_name} (R²: {results[best_name]['r2']:.4f})")

    model_path = MODELS_DIR / "marketing_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"Modèle sauvegardé : {model_path}")

    return best_model, results


if __name__ == "__main__":
    train_model()
