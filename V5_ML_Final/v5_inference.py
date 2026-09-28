
from catboost import CatBoostClassifier
import pandas as pd

MODEL_PATH = "/content/drive/MyDrive/V5_ML_Final/v5_model.cbm"

FEATURES = [
    "name_exact",
    "address_exact",
    "country_exact",
    "name_similarity",
    "address_similarity",
    "name_token_overlap",
    "address_token_overlap",
    "name_jaccard",
    "address_jaccard",
    "shared_name_token_count",
    "shared_address_token_count",
    "shared_address_number_count",
    "name_length_diff",
    "address_length_diff",
    "name_length_ratio",
    "address_length_ratio",
    "similarity_gap",
    "max_similarity",
    "similarity_product",
    "token_overlap_gap"
]

model = CatBoostClassifier()
model.load_model(MODEL_PATH)

def add_v5_features(df):
    X = df[FEATURES[:16]].copy()

    X["similarity_gap"] = (
        X["name_similarity"] - X["address_similarity"]
    ).abs()

    X["max_similarity"] = X[
        ["name_similarity", "address_similarity"]
    ].max(axis=1)

    X["similarity_product"] = (
        X["name_similarity"] * X["address_similarity"]
    )

    X["token_overlap_gap"] = (
        X["name_token_overlap"] - X["address_token_overlap"]
    ).abs()

    return X[FEATURES]

def predict_v5(candidate_features, threshold=0.60):
    X = add_v5_features(candidate_features)
    probability = model.predict_proba(X)[:, 1]
    prediction = (probability >= threshold).astype(int)
    return probability, prediction
