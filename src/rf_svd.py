import pandas as pd
from sklearn import decomposition
from sklearn import ensemble
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics


def run_training(fold):
    df = pd.read_csv("../inputs/train_folds.csv")
    df.review = df.review.apply(str)
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    tfv = TfidfVectorizer()
    tfv.fit(df_train.review.values)

    xtrain = tfv.transform(df_train.review.values)
    xvalid = tfv.transform(df_valid.review.values)

    svd = decomposition.TruncatedSVD(n_components=120)
    svd.fit(xtrain)
    xtrain_svd = svd.transform(xtrain)
    xvalid_svd = svd.transform(xvalid)

    ytrain = df_train.sentiment.values
    yvalid = df_valid.sentiment.values

    rf = ensemble.RandomForestClassifier(
        n_estimators=700,
        n_jobs=-1
    )
    rf.fit(xtrain_svd, ytrain)
    pred = rf.predict_proba(xvalid_svd)[:, 1]

    auc = metrics.roc_auc_score(yvalid, pred)
    print(f'FOLD : {fold}, AUC : {auc}')

    df_valid.loc[:, "lr_svd_pred"] = pred

    return df_valid[["id", "sentiment", "kfold", "lr_svd_pred"]]


if __name__ == "__main__":
    dfs = []
    for j in range(5):
        temp_df = run_training(j)
        dfs.append(temp_df)

    final_valid_df = pd.concat(dfs)
    print(final_valid_df.shape)
    final_valid_df.to_csv("../model_preds/lr_svd_pred.csv", index=False)

    """
FOLD : 0, AUC : 0.88798
FOLD : 1, AUC : 0.8931492
FOLD : 2, AUC : 0.89796672
FOLD : 3, AUC : 0.8903486399999999
FOLD : 4, AUC : 0.8793392000000001


FOLD : 0, AUC : 0.88934192
FOLD : 1, AUC : 0.89154848
FOLD : 2, AUC : 0.89351112
FOLD : 3, AUC : 0.88838928
FOLD : 4, AUC : 0.87884296

FOLD : 0, AUC : 0.87831544
FOLD : 1, AUC : 0.88553536
FOLD : 2, AUC : 0.8804568799999999
FOLD : 3, AUC : 0.8729932
FOLD : 4, AUC : 0.8718794400000001
    """

