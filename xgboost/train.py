import pandas as pd
from preprocess import load_and_preprocess
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

def train(start_year: int = 1968, end_year: int = 2023):
    paths = []
    for year in range(start_year, end_year):
        paths.append(f"../data/atp_matches_{year}.csv")

    df = load_and_preprocess(paths)
    df["y_reg"] = np.where(df["outcome"] == 1, 1.0, 0.0)

    X = df[["playerA_rank", "playerB_rank", "playerA_left_hand", "playerB_left_hand", "age_diff", "height_diff", "elo_diff"]] 
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
        random_state=42
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



if __name__ == "__main__":
    train()