# flake8: noqa
import itertools

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
# import statsmodels.api
from imblearn.over_sampling import SMOTE
from modeling import (fit_logistic_regression, fit_model, get_accuracy,
                      get_precision, get_recall)
from preprocessing import pickle_dump, pickle_load
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler

# import os


def is_continuous(df, column_name):
    # check to see if column is int or float and has more than 2 unique values
    if (
        df[column_name].apply(isinstance, args=[(int, float)]).all()
        and df[column_name].nunique() > 5
    ):
        return True

    # else, column is not continuous
    elif df[column_name].apply(isinstance, args=[(str)]).all():
        return False
    else:
        return False


def cont_cont_heatmap(continuous, dataframe):
    result = []
    for i in continuous:
        holder = []
        for j in continuous:
            holder.append(
                np.round(stats.pearsonr(dataframe[i].values, dataframe[j].values)[0], 3)
            )
        result.append(holder)

    print(result)
    fig = ff.create_annotated_heatmap(
        result, x=continuous, y=continuous, showscale=True, colorscale="Blues"
    )
    fig.update_layout(title="Continuous-Continuous Correlation Matrix")
    fig.show()


# def is_continuous(df, column_name):
#     # check to see if column is int or float and has more than 2 unique values
#     if (
#         df[column_name].apply(isinstance, args=[(int, float)]).all()
#         and df[column_name].nunique() > 5
#     ):
#         return True
#
#     # else, column is not continuous
#     elif df[column_name].apply(isinstance, args=[(str)]).all():
#         return False
#     else:
#         return False


def plot_continuous(df, series, response):
    # if response is continuous, scatterplot with trend line
    if is_continuous(df, response):
        fig = px.scatter(df, x=series.name, y=response, trendline="ols")

    # if response is categorical, violin plot
    else:
        fig = px.violin(df, x=response, y=series.name, box=True)
    return fig


def cont_cont_correlation(dataframe, list_of_predictors, response):
    # os.mkdir("finalTables")
    cont_combinations = list(itertools.combinations(list_of_predictors, 2))
    metrics = []
    pred1 = []
    pred2 = []
    for i in cont_combinations:
        pred1.append(i[0])
        pred2.append(i[1])
        metrics.append(stats.pearsonr(dataframe[i[0]], dataframe[i[1]])[0])

    # create plots for all predictors
    for i in list_of_predictors:
        fig = plot_continuous(dataframe, dataframe[i], response)
        fig.write_html("finalTables/" + i.replace(" ", "") + ".html")

    result = pd.DataFrame(
        list(zip(pred1, pred2, metrics)),
        columns=["predictor1", "predictor2", "pearson coefficient"],
    ).sort_values(by="pearson coefficient", ascending=False)
    result.to_html(
        "finalTables/continuous_predictors_table.html",
        formatters={
            "predictor1": lambda x: f'<a href="{x.replace(" ", "") + ".html"}">{x}</a>',
            "predictor2": lambda x: f'<a href="{x.replace(" ", "") + ".html"}">{x}</a>',
        },
        escape=False,
    )
    return result


def main():

    # first, examine features then run correlation values
    path = "/Volumes/Transcend/DownscalingData/dataModels/final_df"
    df = pickle_load(path)
    conditions = [df["precipitation"] == 0, df["precipitation"] > 0]
    outputs = [0, 1]
    res = np.select(conditions, outputs)
    df["precip_binary"] = pd.Series(res)

    df = df.dropna()
    # pickle_dump(df, path)
    #
    # result = cont_cont_correlation(df, df.columns[4:],"precipitation")
    corrs = pickle_load(
        "/Volumes/Transcend/DownscalingData/dataModels/correlationTable"
    )

    corrs = corrs.dropna()
    highly_correlated = corrs[abs(corrs["pearson coefficient"]) > 0.7]

    drop = []
    keep = ["ps2", "tas2", "rlus2", "tas3", "ps3", "rlus3", "hfls3", "rlds2", "ps1"]
    for i in highly_correlated["predictor2"]:
        if i in keep:
            continue
        else:
            drop.append(i)

    print("Dropping: ", drop)

    df_features_removed = df.drop(columns=drop)

    # result = cont_cont_correlation(df_features_removed, df_features_removed.columns[6:], "precipitation")

    df_features_removed.drop(
        columns=["lat1", "lon1", "lat2", "lon2", "lat3", "lon3", "lat4", "lon4"],
        inplace=True,
    )
    df_features_removed["rainySeason"] = np.where(
        (df_features_removed["Datetime"].dt.month >= 11)
        & (df_features_removed["Datetime"].dt.month < 3),
        1,
        0,
    )
    # print(df_features_removed.columns[2:len(df_features_removed.columns)])
    # Start modeling pipeline
    predictors = []
    for i in df_features_removed.columns[
        2 : len(df_features_removed.columns)
    ]:  # noqa: E731
        if i.lower().startswith("lat") or i.lower().startswith("lon"):
            continue
        predictors.append(i)

    df_features_removed = df_features_removed.sort_values(by=["Datetime"])
    predictors.append("Datetime")
    predictors.append("precip_binary")
    feature_df = df_features_removed.loc[
        :, df_features_removed.columns.isin(predictors)
    ]
    for i in feature_df.columns:
        if i.startswith("clt"):
            feature_df[i] = np.round(feature_df[i])

    # create dummy variables for categoricals
    feature_df = pd.get_dummies(feature_df, columns=["clt1", "clt2", "clt3", "clt4"])

    # split data to train and testing data
    feature_df_train = feature_df[feature_df["Datetime"] < "2006-1-1"]
    # feature_df_test = feature_df[feature_df["Datetime"] > "2006-1-1"]

    x_train = feature_df_train.loc[
        :,
        ~feature_df_train.columns.isin(["Datetime", "precip_binary", "precipitation"]),
    ]
    y_train = feature_df_train["precip_binary"]
    # x_test = feature_df_test.loc[
    # :, ~feature_df_test.columns.isin(["Datetime", "precip_binary", "precipitation"])
    # ]
    # y_test = feature_df_test["precip_binary"]

    scale_columns = []

    for i in x_train.columns:
        if i.startswith("rainy"):
            break
        scale_columns.append(i)

    scaler = MinMaxScaler()
    x_train[scale_columns] = scaler.fit_transform(x_train[scale_columns])

    # use SMOTE to fix class imabalnce
    oversample = SMOTE()
    print("Smote")
    x_train, y_train = oversample.fit_resample(x_train, y_train)

    print("fitting logit Model")
    print("Predictors: ", x_train.columns)
    result = fit_logistic_regression(x_train.values, y_train.values)
    pickle_dump(
        result,
        "/Volumes/Transcend/DownscalingData/dataModels/logit_result_feature_selection1",
    )
    result = pickle_load(
        "/Volumes/Transcend/DownscalingData/dataModels/logit_result_feature_selection1"
    )
    print(
        np.mean(
            get_accuracy(
                result["test_pred_dry_actual_dry"],
                result["test_pred_wet_actual_wet"],
                result["test_pred_wet_actual_dry"],
                result["test_pred_dry_actual_wet"],
            )
        )
    )
    print(
        "Precision: ",
        get_precision(
            result["test_pred_wet_actual_wet"], result["test_pred_wet_actual_dry"]
        ),
    )
    print(
        "Recall: ",
        get_recall(
            result["test_pred_wet_actual_wet"], result["test_pred_dry_actual_wet"]
        ),
    )
    print("fitting rf model")
    clf = RandomForestClassifier(verbose=100, n_jobs=-1)
    res = fit_model(clf, x_train.values, y_train.values)
    pickle_dump(
        res,
        "/Volumes/Transcend/DownscalingData/dataModels/rf_result_feature_selection1",
    )
    print(
        np.mean(
            get_accuracy(
                res["test_pred_dry_actual_dry"],
                res["test_pred_wet_actual_wet"],
                res["test_pred_wet_actual_dry"],
                res["test_pred_dry_actual_wet"],
            )
        )
    )

    print(
        "Precision: ",
        get_precision(res["test_pred_wet_actual_wet"], res["test_pred_wet_actual_dry"]),
    )
    print(
        "Recall: ",
        get_recall(res["test_pred_wet_actual_wet"], res["test_pred_dry_actual_wet"]),
    )


if __name__ == "__main__":
    main()
