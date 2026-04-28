from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd 

df = pd.read_csv('../../data/Dummy Data HSS.csv')

df = df.dropna(subset=["Sales"])

x = df.drop("Sales", axis=1)
y = df["Sales"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

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

x_train_preprocessed = preprocessor.fit_transform(x_train)
x_test_preprocessed = preprocessor.transform(x_test)

print(x_train_preprocessed.shape)
print(x_test_preprocessed.shape)

def run_eda(df):
    correlations = df[["TV", "Radio", "Social Media", "Sales"]].corr()
    print(correlations["Sales"])

run_eda(df)