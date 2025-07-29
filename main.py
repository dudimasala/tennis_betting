import pandas as pd

url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2021.csv"
df = pd.read_csv(url)

print(df.columns)