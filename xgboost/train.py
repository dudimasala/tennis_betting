import pandas as pd
from preprocess import load, add_elos, preprocess, add_recent_form, add_head_to_head
import numpy as np
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import json

# predict_cols = ["rank_diff", "playerA_left_hand", "playerB_left_hand", "age_diff", "height_diff", "elo_diff", "form_diff", "surface"]
predict_cols = ["elo_diff", "form_diff", "rank_diff", "h2h_diff", "total_matches"]

def train(start_year: int = 1968, end_year: int = 2020, fut_start_year: int = 1991, fut_end_year: int = 2024, chl_start_year: int = 1978, chl_end_year: int = 2024):
    paths = []
    # paths.append(f"../data/atp_matches_2023.csv")
    for year in range(start_year, end_year + 1):
        paths.append(f"../data/atp_matches_{year}.csv")

    for year in range(fut_start_year, fut_end_year + 1):
        paths.append(f"../data/atp_matches_futures_{year}.csv")     

    for year in range(chl_start_year, chl_end_year + 1):
        paths.append(f"../data/atp_matches_qual_chall_{year}.csv")      

    df, ratings = add_elos(load(paths))
    with open('elos.json', 'w') as f:
        json.dump(ratings, f)
    
    df = add_recent_form(df)
    df, h2h = add_head_to_head(df)

    h2h_serializable = {
    f"{winner},{loser}": counts
    for (winner, loser), counts in h2h.items()
    }
    with open('h2h.json','w') as f:
        json.dump(h2h_serializable, f)

    df = preprocess(df)
    df["y_reg"] = np.where(df["outcome"] == 1, 1.0, 0.0)
    X = df[predict_cols] 
    y = df["y_reg"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42, 
        enable_categorical=True
    )
    model.fit(X_train, y_train)
    model.save_model("tennis_xgb.json")

    # Eval (reg) - if we are trying to calculate probabilities
    # y_pred = model.predict(X_test)
    # mse = mean_squared_error(y_test, y_pred)
    # rmse = np.sqrt(mse)
    # print(f"Test RMSE: {rmse:.3f}")

    # Eval (cls)
    y_pred_label = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_label)
    classification_error = 1 - accuracy
    print(f"Test Accuracy: {accuracy:.3f}")
    print(f"Classification Error: {classification_error:.3f}")

    importance_dict = model.get_booster().get_score(importance_type='gain')
    plot_importance(importance_dict, importance_type='gain', max_num_features=10)
    plt.title("Top‑10 Feature Importances (by gain)")
    plt.show()



if __name__ == "__main__":
    train()