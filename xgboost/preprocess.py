import pandas as pd
from collections import defaultdict

def preprocess_matches(df: pd.DataFrame) -> pd.DataFrame:
  print(df.columns)
  # 0 is right, 1 is left
  df["winner_left_hand"] = (df.loc[:, "winner_hand"] == 'L')
  df["loser_left_hand"] = (df.loc[:, "loser_hand" ] == 'L')

  winner_cols = [c for c in df.columns if c.startswith("winner_")]
  w_cols = [c for c in df.columns if c.startswith("w_")]
  loser_cols = [c for c in df.columns if c.startswith("loser_")]
  l_cols = [c for c in df.columns if c.startswith("l_")]

  df_A = df[winner_cols + w_cols].copy()
  df_A.columns = [c.replace("winner_", "playerA_") for c in winner_cols] + [c.replace("w_", "playerA_") for c in w_cols]
  df_A["outcome"] = 1
  for c in loser_cols:
    df_A[c.replace("loser_", "playerB_")] = df[c].values
  for c in l_cols:
    df_A[c.replace("l_", "playerB_")] = df[c].values
  
  df_B = df[loser_cols + l_cols].copy()
  df_B.columns = [c.replace("loser_", "playerA_") for c in loser_cols]+ [c.replace("l_", "playerA_") for c in l_cols]
  df_B["outcome"] = 0
  for c in winner_cols:
    df_B[c.replace("winner_", "playerB_")] = df[c].values
  for c in w_cols:
    df_B[c.replace("w_", "playerB_")] = df[c].values

  df_model = pd.concat([df_A, df_B], ignore_index=True)
  df_model = df_model.sample(frac=1, random_state=42).reset_index(drop=True) # shuffle
  df_model["age_diff"] = df_model["playerA_age"] - df_model["playerB_age"]
  df_model["height_diff"] = df_model["playerA_ht"] - df_model["playerB_ht"]
  df_model["elo_diff"] = df_model["playerA_elo"] - df_model["playerB_elo"]
  return df_model

def load_and_preprocess(paths: list[str]) -> pd.DataFrame:
  dfs = [pd.read_csv(path) for path in paths] 
  df = pd.concat(dfs, ignore_index=True)
  df = add_elo(df)
  return preprocess_matches(df)

def add_elo(df: pd.DataFrame, k: int = 32, initial_rating: int = 1500) -> pd.DataFrame:
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

  return df
