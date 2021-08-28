import glob
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import xgboost as xgb

def run_training(pred_df, fold):
    train_df = pred_df[pred_df.kfold != fold].reset_index(drop=True)
    valid_df = pred_df[pred_df.kfold == fold].reset_index(drop=True)

    xtrain = train_df[['lr_svd_pred', 'lr_cnt_pred', 'lr_pred']].values
    xvalid = valid_df[['lr_svd_pred', 'lr_cnt_pred', 'lr_pred']].values

    Std = StandardScaler()
    xtrain = Std.fit_transform(xtrain)
    xvalid = Std.transform(xvalid)
    Lr = xgb.XGBClassifier()
    Lr.fit(xtrain, train_df.sentiment.values)
    preds = Lr.predict_proba(xvalid)[:, 1]
    auc = metrics.roc_auc_score(valid_df.sentiment.values, preds)
    print(f"FOLD : {fold}, AUC : {auc}")
    valid_df.loc[:, "xgb_preds"] = preds
    return valid_df


if __name__ == "__main__":
    files = glob.glob("../model_preds/*.csv")
    df = None
    for f in files:
        if df is None:
            df = pd.read_csv(f)
        else:
            temp_df = pd.read_csv(f)
            df = df.merge(temp_df, on="id", how="left")
    target = df.sentiment.values
    pred_cols = ['lr_svd_pred', 'lr_cnt_pred', 'lr_pred']
    dfs = []
    for j in range(5):
        temp_df = run_training(df, j)
        dfs.append(temp_df)

    final_valid_df = pd.concat(dfs)
    print(metrics.roc_auc_score(final_valid_df.sentiment.values, final_valid_df.xgb_preds.values))