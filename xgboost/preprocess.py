import pandas as pd
from typing import Tuple, Dict, Union
from collections import defaultdict


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
  # 0 is right, 1 is left
  df["winner_left_hand"] = (df.loc[:, "winner_hand"] == 'L')
  df["loser_left_hand"] = (df.loc[:, "loser_hand" ] == 'L')

  winner_cols = [c for c in df.columns if c.startswith("winner_")]
  w_cols = [c for c in df.columns if c.startswith("w_")]
  loser_cols = [c for c in df.columns if c.startswith("loser_")]
  l_cols = [c for c in df.columns if c.startswith("l_")]
  other_cols = [c for c in df.columns if c not in winner_cols + w_cols + loser_cols + l_cols]

  df_A = df[winner_cols + w_cols + other_cols].copy()
  df_A.columns = [c.replace("winner_", "playerA_") for c in winner_cols] + [c.replace("w_", "playerA_") for c in w_cols] + other_cols
  # 0 if playerA is the winner, 1 if playerB is the winner
  df_A["outcome"] = 0
  for c in loser_cols:
    df_A[c.replace("loser_", "playerB_")] = df[c].values
  for c in l_cols:
    df_A[c.replace("l_", "playerB_")] = df[c].values
  
  df_B = df[loser_cols + l_cols + other_cols].copy()
  df_B.columns = [c.replace("loser_", "playerA_") for c in loser_cols]+ [c.replace("l_", "playerA_") for c in l_cols] + other_cols
  df_B["outcome"] = 1
  for c in winner_cols:
    df_B[c.replace("winner_", "playerB_")] = df[c].values
  for c in w_cols:
    df_B[c.replace("w_", "playerB_")] = df[c].values

  df_model = pd.concat([df_A, df_B], ignore_index=True)
  df_model = df_model.sample(frac=1, random_state=42).reset_index(drop=True) # shuffle
  df_model["age_diff"] = df_model["playerA_age"] - df_model["playerB_age"]
  df_model["height_diff"] = df_model["playerA_ht"] - df_model["playerB_ht"]
  df_model["elo_diff"] = df_model["playerA_elo"] - df_model["playerB_elo"]
  df_model["rank_diff"] = df_model["playerA_rank"] - df_model["playerB_rank"]
  df_model["form_diff"] = df_model["playerA_form"] - df_model["playerB_form"]
  df_model["surface"] = df_model["surface"].astype("category")
  df_model["h2h_diff"] = df_model["playerA_wins"] - df_model["playerB_wins"]


  return df_model

def load(paths: list[str]) -> pd.DataFrame:
  dfs = [pd.read_csv(path) for path in paths] 
  return pd.concat(dfs, ignore_index=True)  


def add_recent_form(df: pd.DataFrame, window: int = 10):
  date_col = "tourney_date"
  df_w = df.loc[:, [date_col, "winner_id"]].copy()
  df_l = df.loc[:, [date_col, "loser_id"].copy()]
  df_w["player_id"] = df_w["winner_id"]
  df_l["player_id"] = df_l["loser_id"]
  df_w["outcome"] = 1
  df_l["outcome"] = 0

  cdf = pd.concat([df_w, df_l], ignore_index=True)
  cdf = cdf.sort_values(date_col).reset_index(drop=True)

  cdf["form"] = (cdf.groupby("player_id")["outcome"].transform(lambda s: s.shift().rolling(window, min_periods=1).mean()))
  winner_form = cdf.loc[cdf["outcome"] == 1, ["tourney_date", "player_id", "form"]]
  winner_form.columns = ["tourney_date", "winner_id","winner_form"]
  loser_form = cdf.loc[cdf["outcome"] == 0, ["tourney_date", "player_id", "form"]]
  loser_form.columns = ["tourney_date", "loser_id", "loser_form"]
  df = df.merge(winner_form, on=["tourney_date", "winner_id"], how="left").merge(loser_form, on=["tourney_date", "loser_id"], how="left")
  return df




def add_elos(df: pd.DataFrame, k: int = 32, initial_rating: int = 1500) -> Tuple[pd.DataFrame, Dict[int, float]]:
  date_col = "tourney_date"
  df = df.sort_values(date_col).reset_index(drop=True)
  ratings = defaultdict(lambda: initial_rating)

  eloWinner = []
  eloLoser = []

  for row in df.itertuples(index=False):
    ra = ratings[row.winner_id]
    rb = ratings[row.loser_id]

    # prematch ratings
    eloWinner.append(ra)
    eloLoser.append(rb)

    # expected
    ea = 1 / (1 + 10 ** ((rb - ra) / 400))

    # update ratings
    ratings[row.winner_id] += k * (1 - ea)
    ratings[row.loser_id] += k * (ea - 1)
  
  df["winner_elo"] = eloWinner
  df["loser_elo"] = eloLoser

  return df, ratings


def add_head_to_head(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], Dict]]:
  date_col = "tourney_date"
  df = df.copy()
  df = df.sort_values(date_col).reset_index(drop=True)

  h2h = defaultdict(lambda: {'A': 0, 'B': 0})

  wins_A_list = []
  wins_B_list = []

  for _, row in df.iterrows():
    a = row["winner_id"]
    b = row["loser_id"]
    wins_A = h2h[(a, b)]["A"]
    wins_B = h2h[(a, b)]["B"]
    wins_A_list.append(wins_A)
    wins_B_list.append(wins_B)
    h2h[(a, b)]["A"] += 1
    h2h[(b, a)]["B"] += 1
  
  df["winner_wins"] = wins_A_list
  df["loser_wins"] = wins_B_list
  df["winner_wins"] //= 2
  df["loser_wins"] //= 2
  df["total_matches"] = df["winner_wins"] + df["loser_wins"]

  return df, h2h

