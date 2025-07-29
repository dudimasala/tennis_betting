import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from preprocess import load_and_preprocess

def load_model(path: str, model_type: str = "reg"):
  if model_type == "reg":
      model = XGBRegressor()
  else:
      model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
  model.load_model(path)
  return model


def predict(paths: list[str], model) -> pd.DataFrame:
    df = load_and_preprocess(paths)

    X = df[["playerA_rank", "playerB_rank", "playerA_left_hand", "playerB_left_hand", "age_diff", "height_diff", "elo_diff"]]

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