import glob
import pandas as pd
import numpy as np
from sklearn import metrics

if __name__ == "__main__":
    files = glob.glob("../model_preds/*.csv")
    df = None
    for f in files:
        if df is None:
            df = pd.read_csv(f)
        else:
            temp_df = pd.read_csv(f)
            df = df.merge(temp_df, on="id", how="left")
    # print(df.head(10))
    target = df.sentiment.values
    pred_cols = ['lr_svd_pred', 'lr_cnt_pred', 'lr_pred']

    for col in pred_cols:
        auc = metrics.roc_auc_score(target, df[col].values)
        print(f"{col}, overall_auc = {auc}")

    avg_preds = np.mean(df[['lr_svd_pred', 'lr_cnt_pred', 'lr_pred']].values, axis=1)
    print(f"Average of all model accuracy : {metrics.roc_auc_score(target, avg_preds)}")

    lr_pred = df.lr_pred.values
    lr_cnt_pred = df.lr_cnt_pred.values
    lr_svd_pred = df.lr_svd_pred.values

    avg_pred = (lr_pred + 3 * lr_cnt_pred + lr_svd_pred) / 5
    print(f"Weighted Average accuracy : {metrics.roc_auc_score(target, avg_pred)}")

    print("rank averaging")
    lr_pred = df.lr_pred.rank().values
    lr_cnt_pred = df.lr_cnt_pred.rank().values
    lr_svd_pred = df.lr_svd_pred.rank().values
    avg_pred = (lr_pred + lr_cnt_pred + lr_svd_pred) / 3
    print(metrics.roc_auc_score(target, avg_pred))

    print("rank weighted averaging")
    lr_pred = df.lr_pred.rank().values
    lr_cnt_pred = df.lr_cnt_pred.rank().values
    lr_svd_pred = df.lr_svd_pred.rank().values
    avg_pred = (lr_pred + 3 * lr_cnt_pred + lr_svd_pred) / 5
    print(metrics.roc_auc_score(target, avg_pred))
