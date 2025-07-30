import pandas as pd
from typing import Tuple, Dict, Union, List
from collections import defaultdict
import matplotlib.pyplot as plt


def sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
  return df.sort_values(['tourney_date', 'match_num', 'tourney_id']).reset_index(drop=True)


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

  df_model["match_key"] = df_model.apply(
    lambda r: (
        min(r.playerA_id, r.playerB_id),
        max(r.playerA_id, r.playerB_id),
        r.tourney_id, r.match_num  # or whatever date field you have
    ),
    axis=1
  )

  df_model = df_model.drop_duplicates(subset="match_key", keep="first").reset_index(drop=True)

  df_model["age_diff"] = df_model["playerA_age"] - df_model["playerB_age"]
  df_model["height_diff"] = df_model["playerA_ht"] - df_model["playerB_ht"]
  df_model["elo_diff"] = df_model["playerA_elo"] - df_model["playerB_elo"]
  df_model["hard_elo_diff"] = df_model["playerA_hard_elo"] - df_model["playerB_hard_elo"]
  df_model["clay_elo_diff"] = df_model["playerA_clay_elo"] - df_model["playerB_clay_elo"]
  df_model["grass_elo_diff"] = df_model["playerA_grass_elo"] - df_model["playerB_grass_elo"]
  df_model["carpet_elo_diff"] = df_model["playerA_carpet_elo"] - df_model["playerB_carpet_elo"]
  df_model["rank_diff"] = df_model["playerA_rank"] - df_model["playerB_rank"]
  df_model["form_diff"] = df_model["playerA_form"] - df_model["playerB_form"]
  df_model["surface"] = df_model["surface"].astype("category")
  df_model["h2h_diff"] = df_model["playerA_wins"] - df_model["playerB_wins"]



  return df_model

def load(paths: list[str]) -> pd.DataFrame:
  dfs = [pd.read_csv(path) for path in paths] 
  df = pd.concat(dfs, ignore_index=True) 
  df["tourney_date"] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
  return df


def add_recent_form(df: pd.DataFrame, window: int = 10):
  date_col = "tourney_date"
  df_w = df.loc[:, ["tourney_id", date_col, "match_num", "winner_id"]].copy()
  df_l = df.loc[:, ["tourney_id", date_col, "match_num", "loser_id"].copy()]
  df_w["player_id"] = df_w["winner_id"]
  df_l["player_id"] = df_l["loser_id"]
  df_w["outcome"] = 1
  df_l["outcome"] = 0
# need to fix the sort by date
  cdf = pd.concat([df_w, df_l], ignore_index=True)
  cdf = sort_chronologically(cdf)

  cdf["form"] = (cdf.groupby("player_id")["outcome"].transform(lambda s: s.shift().rolling(window, min_periods=1).mean()))
  winner_form = cdf.loc[cdf["outcome"] == 1, ["tourney_id", "match_num", "player_id", "form"]]
  winner_form.columns = ["tourney_id", "match_num", "winner_id","winner_form"]
  loser_form = cdf.loc[cdf["outcome"] == 0, ["tourney_id", "match_num", "player_id", "form"]]
  loser_form.columns = ["tourney_id", "match_num", "loser_id", "loser_form"]

  df = df.merge(winner_form, on=["tourney_id", "match_num", "winner_id"], how="left")
  df = df.merge(loser_form, on=["tourney_id", "match_num", "loser_id"], how="left")
  return df


def calc_and_update_ratings(ratings: Dict, winner_id: int, loser_id: int, winner_elo: List[float], loser_elo: List[float], update: bool, k: int = 32) -> None:
  ra = ratings[winner_id]
  rb = ratings[loser_id]
  winner_elo.append(ra)
  loser_elo.append(rb)
  if update:
    ea = 1 / (1 + 10 ** ((rb - ra) / 400))
    ratings[winner_id] += k * (1 - ea)
    ratings[loser_id] += k * (ea - 1)



def add_elos(df: pd.DataFrame, k: int = 32, initial_rating: int = 1500) -> Tuple[pd.DataFrame, Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float]]:
  df = sort_chronologically(df)
  ratings = defaultdict(lambda: initial_rating)

  eloWinner = []
  eloLoser = []

  hard_ratings = defaultdict(lambda: initial_rating)
  clay_ratings = defaultdict(lambda: initial_rating)
  grass_ratings = defaultdict(lambda: initial_rating)
  carpet_ratings = defaultdict(lambda: initial_rating)

  hardEloWinner = []
  clayEloWinner = []
  grassEloWinner = []
  carpetEloWinner = []
  hardEloLoser = []
  clayEloLoser = []
  grassEloLoser = []
  carpetEloLoser = []

  surface_map = {
    "Hard":   (hard_ratings,   hardEloWinner,   hardEloLoser),
    "Clay":   (clay_ratings,   clayEloWinner,   clayEloLoser),
    "Grass":  (grass_ratings,  grassEloWinner,  grassEloLoser),
    "Carpet": (carpet_ratings, carpetEloWinner, carpetEloLoser),
  }

  for row in df.itertuples(index=False):
    calc_and_update_ratings(ratings, row.winner_id, row.loser_id, eloWinner, eloLoser, update=True)

    for surf, (surf_ratings, eloW, eloL) in surface_map.items():
      should_update = (row.surface == surf)
      calc_and_update_ratings(surf_ratings, row.winner_id, row.loser_id, eloW, eloL, update=should_update)

  df["winner_elo"] = eloWinner
  df["loser_elo"] = eloLoser
  df["winner_hard_elo"] = hardEloWinner
  df["loser_hard_elo"] = hardEloLoser
  df["winner_clay_elo"] = clayEloWinner
  df["loser_clay_elo"] = clayEloLoser
  df["winner_grass_elo"] = grassEloWinner
  df["loser_grass_elo"] = grassEloLoser
  df["winner_carpet_elo"] = carpetEloWinner
  df["loser_carpet_elo"] = carpetEloLoser

  print(df[["winner_carpet_elo"]])


  return df, ratings, hard_ratings, clay_ratings, grass_ratings, carpet_ratings


def add_head_to_head(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], Dict]]:
  date_col = "tourney_date"
  df = df.copy()
  df = sort_chronologically(df)

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
  df["total_matches"] = df["winner_wins"] + df["loser_wins"]

  return df, h2h

