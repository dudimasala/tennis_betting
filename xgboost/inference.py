import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from preprocess import load, preprocess, add_recent_form
from train import predict_cols
import json
from collections import defaultdict

def ints_object_hook(d):
    # Convert each key and value in this dict to int
    return {int(k): int(v) for k, v in d.items()}

def load_h2h_counts():
    raw = json.load(open('h2h.json'))
    h2h = defaultdict(lambda: {"A":0,"B":0})
    for key, counts in raw.items():
        a_str, b_str = key.split(",")
        h2h[(int(a_str), int(b_str))] = counts
    return h2h

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
    with open('hard_elos.json') as f:
        hard_ratings = json.load(f, object_hook=ints_object_hook)
    with open('clay_elos.json') as f:
        clay_ratings = json.load(f, object_hook=ints_object_hook)
    with open('grass_elos.json') as f:
        grass_ratings = json.load(f, object_hook=ints_object_hook)
    with open('carpet_elos.json') as f:
        carpet_ratings = json.load(f, object_hook=ints_object_hook)

    
    df['winner_elo'] = df['winner_id'].map(ratings)
    df['loser_elo'] = df['loser_id'].map(ratings)
    df['winner_hard_elo'] = df['winner_id'].map(hard_ratings)
    df['loser_hard_elo'] = df['loser_id'].map(hard_ratings)
    df['winner_clay_elo'] = df['winner_id'].map(clay_ratings)
    df['loser_clay_elo'] = df['loser_id'].map(clay_ratings)
    df['winner_grass_elo'] = df['winner_id'].map(grass_ratings)
    df['loser_grass_elo'] = df['loser_id'].map(grass_ratings)
    df['winner_carpet_elo'] = df['winner_id'].map(carpet_ratings)
    df['loser_carpet_elo'] = df['loser_id'].map(carpet_ratings)

    df = add_recent_form(df)

    h2h_counts = load_h2h_counts()
    winsA = { pair: counts["A"] for pair, counts in h2h_counts.items() }
    winsB = { pair: counts["B"] for pair, counts in h2h_counts.items() }

    winsA_s = pd.Series(winsA)
    winsB_s = pd.Series(winsB)

    df["pair"] = list(zip(df["winner_id"], df["loser_id"]))
    df["winner_wins"] = df["pair"].map(winsA_s).fillna(0).astype(int)
    df["loser_wins" ] = df["pair"].map(winsB_s).fillna(0).astype(int)
    df.drop(columns="pair", inplace=True)
    df["total_matches"] = df["winner_wins"] + df["loser_wins"]

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
    paths = ["../data/atp_matches_2024.csv"]
    df_pred = predict(paths, model)

    if "pred_reg" in df_pred.columns:
      df_pred["correct"] = (
          (df_pred["pred_reg"] >= 0.5) & (df_pred["outcome"] == 1)
      ) | (
          (df_pred["pred_reg"] < 0.5) & (df_pred["outcome"] == 0)
      )
    else:
      df_pred["correct"] = (df_pred["pred_label"] == df_pred["outcome"])

    print(df_pred["correct"].sum() / len(df_pred))
    print(df_pred[["playerA_name", "playerB_name", "playerA_elo", "playerB_elo", "h2h_diff", "total_matches", "pred_label", "outcome"]])
