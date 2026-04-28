from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "Dummy Data HSS.csv"
MODELS_DIR = PROJECT_ROOT / "src" / "models"


def load_data(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["Sales"])
    return df


def get_preprocessing_pipeline() -> ColumnTransformer:
    numeric_features = ["TV", "Radio", "Social Media"]
    categorical_features = ["Influencer"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])


def get_preprocessed_data(
    filepath: Path = DATA_PATH,
    save_preprocessor: bool = False,
    preprocessor_path: Path = MODELS_DIR / "preprocessor.joblib",
):
    df = load_data(filepath)

    x = df.drop("Sales", axis=1)
    y = df["Sales"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    preprocessor = get_preprocessing_pipeline()
    x_train_preprocessed = preprocessor.fit_transform(x_train)
    x_test_preprocessed = preprocessor.transform(x_test)

    if save_preprocessor:
        preprocessor_path = Path(preprocessor_path)
        preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, preprocessor_path)
        print(f"Preprocessor saved to {preprocessor_path}")

    return x_train_preprocessed, x_test_preprocessed, y_train, y_test, preprocessor


if __name__ == "__main__":
    x_train, x_test, y_train, y_test, preprocessor = get_preprocessed_data(
        save_preprocessor=True
    )

    print(f"x_train shape : {x_train.shape}")
    print(f"x_test shape  : {x_test.shape}")
    print(f"y_train shape : {y_train.shape}")
    print(f"y_test shape  : {y_test.shape}")
