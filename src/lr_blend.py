import glob
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
def run_training(pred_df, fold):
    train_df = pred_df[pred_df.kfold != fold].reset_index(drop=True)
    valid_df = pred_df[pred_df.kfold == fold].reset_index(drop=True)

    xtrain = train_df[['lr_svd_pred', 'lr_cnt_pred', 'lr_pred']].values
    xvalid = valid_df[['lr_svd_pred', 'lr_cnt_pred', 'lr_pred']].values

    Std = StandardScaler()
    xtrain = Std.fit_transform(xtrain)
    xvalid = Std.transform(xvalid)
    Lr = LinearRegression()
    Lr.fit(xtrain, train_df.sentiment.values)
    preds = Lr.predict(xvalid)
    auc = metrics.roc_auc_score(valid_df.sentiment.values, preds)
    print(f"FOLD : {fold}, AUC : {auc}")
    return Lr.coef_


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
    coefs = []
    for j in range(5):
        coefs.append(run_training(df, j))

    coefs = np.array(coefs)
    print(coefs)
    coefs = np.mean(coefs, axis=0)
    print(coefs)

    wgt_average = (
        coefs[0] * df.lr_svd_pred.values +
        coefs[1] * df.lr_cnt_pred.values +
        coefs[2] * df.lr_pred.values
    )
    print("optimial_auc_after_coefs")
    print(metrics.roc_auc_score(target, wgt_average))

# rank weighted averaging
# 0.9516804032

# optimial_auc_after_coefs
# 0.950715744

"""After using LogisticRegression """
#[0.51974779 1.36824978 0.9789071 ]
# optimial_auc_after_coefs
# 0.9506547519999999

"""After using LinearRegression"""
# [0.03506842 0.26973167 0.11795882]
# optimial_auc_after_coefs
# 0.950715744 
# little bit improvement after using LineraRegression