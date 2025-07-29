import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from preprocess import load, preprocess
from train import predict_cols
import json

def ints_object_hook(d):
    # Convert each key and value in this dict to int
    return {int(k): int(v) for k, v in d.items()}

def load_model(path: str, model_type: str = "reg"):
  if model_type == "reg":
      model = XGBRegressor()
  else:
      model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
  model.load_model(path)
  return model


def predict(paths: list[str], model) -> pd.DataFrame:
    df = load(paths)
    # calc elos
    with open('elos.json') as f:
        ratings = json.load(f, object_hook=ints_object_hook)
    
    df['winner_elo'] = df['winner_id'].map(ratings)
    df['loser_elo'] = df['loser_id'].map(ratings)


    df = preprocess(df)

    X = df[predict_cols]

    if hasattr(model, "predict_proba"):
        # classifier
        df["pred_proba"] = model.predict_proba(X)[:, 1]
        df["pred_label"] = model.predict(X)
    else:
        df["pred_reg"] = model.predict(X)

    return df

if __name__ == "__main__":
    MODEL_PATH = "tennis_xgb.json"
    model = load_model(MODEL_PATH, model_type="cls")
    path = "../data/atp_matches_2024.csv"
    df_pred = predict([path], model)

    if "pred_reg" in df_pred.columns:
      df_pred["correct"] = (
          (df_pred["pred_reg"] >= 0.5) & (df_pred["outcome"] == 1)
      ) | (
          (df_pred["pred_reg"] < 0.5) & (df_pred["outcome"] == 0)
      )
    else:
      df_pred["correct"] = (df_pred["pred_label"] == df_pred["outcome"])

    print(df_pred["correct"].sum() / len(df_pred))
    print(df_pred[["playerA_name", "playerB_name", "playerA_elo", "playerB_elo", "pred_label", "outcome"]])
