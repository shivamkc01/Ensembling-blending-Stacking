import glob
import pandas as pd
import numpy as np
from sklearn import metrics
from scipy.optimize import fmin
from functools import partial

class OptimizeAUC:
    def __init__(self):
        self.coef_ = 0

    def _auc(self, coef, X, y):
        x_coef = X * coef
        predictions = np.sum(x_coef, axis=1)
        auc_score = metrics.roc_auc_score(y, predictions)
        # remember to return negative auc because we need to minimize the auc
        return -1.0 * auc_score

    def fit(self, X, y):
        pratial_loss = partial(self._auc, X=X, y=y)
        init_coef = np.random.dirichlet(np.ones(X.shape[1]))
        self.coef_ = fmin(pratial_loss, init_coef, disp=True)

    def predict(self, X):
        x_coef = X * self.coef_
        predictions = np.sum(x_coef, axis=1)
        return predictions

def run_training(pred_df, fold):
    train_df = pred_df[pred_df.kfold != fold].reset_index(drop=True)
    valid_df = pred_df[pred_df.kfold == fold].reset_index(drop=True)

    xtrain = train_df[['lr_svd_pred', 'lr_cnt_pred', 'lr_pred']].values
    xvalid = valid_df[['lr_svd_pred', 'lr_cnt_pred', 'lr_pred']].values

    otp = OptimizeAUC()
    otp.fit(xtrain, train_df.sentiment.values)
    preds = otp.predict(xvalid)
    auc = metrics.roc_auc_score(valid_df.sentiment.values, preds)
    print(f"FOLD : {fold}, AUC : {auc}")
    return otp.coef_


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
    print("optimial_auc")
    print(metrics.roc_auc_score(target, wgt_average))

# rank weighted averaging
# 0.9516804032