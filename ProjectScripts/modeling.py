# flake8: noqa
import numpy as np
import pandas as pd
# import plotly.express as px
# from imblearn.over_sampling import SMOTE
from preprocessing import pickle_load
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler


def confusion_matrix_scorer(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return {
        "pred_dry_actual_dry": cm[0, 0],
        "pred_wet_actual_dry": cm[0, 1],
        "pred_dry_actual_wet": cm[1, 0],
        "pred_wet_actual_wet": cm[1, 1],
    }


def fit_logistic_regression(x_train, y_train, cv=5):
    log_model = LogisticRegression(n_jobs=-1)
    # scoring = {'acc': 'accuracy',
    #            'prec_macro': 'precision_macro',
    #            'rec_micro': 'recall_macro'}
    scores = cross_validate(
        log_model, x_train, y_train, cv=cv, scoring=confusion_matrix_scorer
    )
    print(scores.keys())
    print(
        scores["test_pred_dry_actual_dry"],
        scores["test_pred_dry_actual_wet"],
        scores["test_pred_wet_actual_wet"],
        scores["test_pred_wet_actual_dry"],
    )
    return scores


def fit_model(model, x_train, y_train, cv=5):
    scores = cross_validate(
        model, x_train, y_train, cv=cv, scoring=confusion_matrix_scorer
    )
    model.fit(x_train, y_train)
    return scores


def get_accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)


def get_precision(tp, fp):
    return (tp) / (tp + fp)


def get_recall(tp, fn):
    return (tp) / (tp + fn)


def main():
    path = "/Volumes/Transcend/DownscalingData/dataModels/final_df"
    df = pickle_load(path)

    # 2 step modeling:
    # 1) predict dry or wet period (1 or 0) classification
    # 2) if wet day (1), then predict amount of precipitation in millimeters (regression)
    # can also do extreme rain day vs light rain day (classification) then regress on that

    # 1) Classification: wet(1) vs dry(0) period, first, visualize wet vs dry periods
    print(df["precip_binary"].value_counts() / len(df))

    predictors = []
    for i in df.columns[2 : len(df.columns) - 2]:  # noqa: E731
        if i.lower().startswith("lat") or i.lower().startswith("lon"):
            continue
        predictors.append(i)

    # first, sort original data by date in order to preserve time order
    df = df.sort_values(by=["Datetime"])
    predictors.append("Datetime")
    predictors.append("precip_binary")
    feature_df = df.loc[:, df.columns.isin(predictors)]
    for i in feature_df.columns:
        if i.startswith("clt"):
            feature_df[i] = np.round(feature_df[i])

    feature_df = pd.get_dummies(feature_df, columns=["clt1", "clt2", "clt3", "clt4"])
    # for i in feature_df.columns:
    #     if i.endswith("2"):
    #         break
    #     #px.histogram(feature_df, feature_df[i]).show()

    # first, do a train/test split.
    # because the data is time series, need to preserve time so cannot be a random split
    # use 1980 - 2005 as training data, then 2006 - 2012 as testing data, so split these dataframes first
    feature_df_train = feature_df[feature_df["Datetime"] < "2006-1-1"]
    # feature_df_test = feature_df[feature_df["Datetime"] > "2006-1-1"]

    # now get x_train, y_train, x_test, and y_test variables
    x_train = feature_df_train.loc[
        :, ~feature_df_train.columns.isin(["Datetime", "precip_binary"])
    ]
    # y_train = feature_df_train["precip_binary"]
    # x_test = feature_df_test.loc[
    #     :, ~feature_df_test.columns.isin(["Datetime", "precip_binary"])
    # ]
    # y_test = feature_df_test["precip_binary"]

    scale_columns = []
    for i in x_train.columns:
        if i.startswith("clt"):
            break
        scale_columns.append(i)
    # print(scale_columns)

    # fit a min max scaler on the training data to normalize all of the predictor values
    scaler = MinMaxScaler()
    x_train[scale_columns] = scaler.fit_transform(x_train[scale_columns])

    # use SMOTE to fix class imabalnce
    # oversample = SMOTE()
    # x_train, y_train = oversample.fit_resample(x_train, y_train)
    # print(x_train.columns)
    # print(y_train.value_counts())
    # result2 = fit_logistic_regression(x_train.values, y_train.values)
    # pickle_dump(result2, "/Volumes/Transcend/DownscalingData/dataModels/logit_result_0.7ratio")
    # result = pickle_load("/Volumes/Transcend/DownscalingData/dataModels/logit_result")
    # result2 = pickle_load("/Volumes/Transcend/DownscalingData/dataModels/logit_result_0.7ratio")

    # print(np.mean(get_accuracy(result['test_pred_dry_actual_dry'], result['test_pred_wet_actual_wet'],
    #                    result['test_pred_wet_actual_dry'], result['test_pred_dry_actual_wet'])))
    #
    # print("recall: ", get_recall(result["test_pred_wet_actual_wet"], result["test_pred_dry_actual_wet"]))
    # print('Precision: ', get_precision(result["test_pred_wet_actual_wet"],result["test_pred_wet_actual_dry"]))

    print("Random Forest")
    clf = RandomForestClassifier(n_jobs=-1, verbose=100)
    # res = fit_model(clf, x_train.values, y_train.values)
    # pickle_dump(res,"/Volumes/Transcend/DownscalingData/dataModels/rf_result")
    #
    # print(np.mean(get_accuracy(res['test_pred_dry_actual_dry'], res['test_pred_wet_actual_wet'],
    #                            res['test_pred_wet_actual_dry'], res['test_pred_dry_actual_wet'])))
    #
    # print("recall: ", np.mean(get_recall(res["test_pred_wet_actual_wet"], res["test_pred_dry_actual_wet"])))
    # print('Precision: ',np.mean(get_precision(res["test_pred_wet_actual_wet"], res["test_pred_wet_actual_dry"])))
    # clf.fit(x_train.values, y_train.values)
    # pd.to_pickle(clf, "/Volumes/Transcend/DownscalingData/dataModels/rf_fit_all")
    clf = pd.read_pickle("/Volumes/Transcend/DownscalingData/dataModels/rf_fit_all")
    importances = list(clf.feature_importances_)
    columns = list(x_train.columns)
    feature_importances_df = pd.DataFrame(
        {"feature": columns, "importance": importances}
    ).sort_values(by="importance", ascending=False)
    print(feature_importances_df.head(20).to_string())


# GAN to generate real samples (maybe instead of smote)


if __name__ == "__main__":
    main()
