import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data(filepath):
    df = pd.read_csv('../../data/Dummy Data HSS.csv')
    df = df.dropna(subset=["Sales"])
    return df

def get_preprocessing_pipeline():
    numeric_features= ["TV", "Radio", "Social Media"]
    categorical_features = ["Influencer"]

    # Gere null autres Sales
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Transforme Influenceur en [0,1]
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])  
    return preprocessor


def get_preprocessed_data(filepath, save_preprocesssor=False, preprocessor_path="preprocessor.joblib"):
    df = load_data(filepath)

    x = df.drop("Sales", axis=1)
    y = df["Sales"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    preprocessor = get_preprocessing_pipeline()

    x_train_preprocessed = preprocessor.fit_transform(x_train)
    x_test_preprocessed = preprocessor.transform(x_test)

    if save_preprocesssor:
        joblib.dump(preprocessor, preprocessor_path)

    return x_train_preprocessed, x_test_preprocessed, y_train, y_test, preprocessor

if __name__ == "__main__":
    filepath = "../../data/Dummy Data HSS.csv"
    x_train, x_test, y_train, y_test, preprocessor = get_preprocessed_data(filepath, save_preprocesssor=True, preprocessor_path="preprocessor.joblib")  

    print("Preprocessing completed. Preprocessor saved to preprocessor.joblib")
    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape) 

