# flake8: noqa
from datetime import datetime

import numpy as np
import pandas as pd
from featurewiz import featurewiz
# from sklearn.model_selection import cross_validate
# from imblearn.over_sampling import SMOTE
from modeling import (fit_model, get_accuracy, get_precision,  # noqa: E731
                      get_recall)
from preprocessing import pickle_dump, pickle_load  # noqa: E731
from sklearn import metrics
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# from scipy import stats
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import MinMaxScaler


def test_score(model, y_test, x_test):
    yhat = model.predict(x_test)
    tn, fp, fn, tp = confusion_matrix(y_test, yhat).ravel()
    acc = get_accuracy(tp, tn, fp, fn)
    print("Precision: ", metrics.precision_score(y_test, yhat, average="weighted"))
    print("Recall: ", metrics.recall_score(y_test, yhat, average="weighted"))
    return acc, confusion_matrix(y_test, yhat)


path = "/Volumes/Transcend/DownscalingData/dataModels/final_df"
df = pickle_load(path)

# create a new column that contains just the date
df["date"] = df["Datetime"].dt.date

# drop old binary column
df = df.drop(columns=["precip_binary"])
df = df.dropna()

# print(df.head().to_string())

# group by date and city, perform aggregation based on column and what makes sense:
# clt: Total cloud fraction -> mode because this is categorical and takes in values: 0, 50, 100
# hfls: "Surface Upward Latent Heat Flux" -> mean because the data is in 3hr mean already
# hfss: "Surface Upward Sensible Heat Flux" -> mean ""
# huss: "Near-Surface Specific Humidity" -> mean bc sampled synoptically
# pr: "precipitation_flux" -> mean
# prc: "convective precipitation_flux" -> mean
# ps: "Surface Pressure" -> mean
# rlds: "Surface Downwelling Longwave Radiation" -> mean
# rlus: "Surface Upwelling Longwave Radiation" -> mean
# rsds: "Surface Downwelling Shortwave Radiation" -> mean
# rsdsdiff: "Surface Diffuse Downwelling Shortwave Radiation" -> mean
# rsus: "Surface Upwelling Shortwave Radiation" -> mean
# tas: "Air Temperature" -> mean
# uas: "Eastward Near-Surface Wind Speed" -> mean
# vas: "Northward Near-Surface Wind Speed" -> mean
# precipitation: -> sum
clt = []
means = []
for i in df.columns:
    if (
        i.lower().startswith("date")
        or i == "City"
        or i == "precipitation"
        or i.lower().startswith("lat")
        or i.lower().startswith("lon")
    ):
        continue
    elif i.startswith("clt"):
        clt.append(i)
    else:
        means.append(i)


aggregations = {}
for i in clt:
    aggregations[i] = "mean"
for j in means:
    aggregations[j] = "mean"

aggregations["precipitation"] = "sum"

df_day = df.groupby(["date", "City"]).agg(aggregations).reset_index()

# create new binary column
conditions = [df_day["precipitation"] == 0, df_day["precipitation"] > 0]
outputs = [0, 1]
res = pd.Series(np.select(conditions, outputs))
df_day["precip_binary"] = res.values


df_day = df_day.sort_values(by=["City", "date"])
print(df_day["precip_binary"].value_counts() / len(df_day))
# print(df_day["clt1"].value_counts())

# first just model with all predictors
predictors = []
for i in df_day.columns:
    if i.lower().startswith("lat") or i.lower().startswith("lon"):
        continue
    else:
        predictors.append(i)


feature_df = df_day.loc[:, df_day.columns.isin(predictors)]

feature_df_train = feature_df[
    feature_df["date"] < datetime.date(datetime.strptime("2006-1-1", "%Y-%m-%d"))
]
feature_df_test = feature_df[
    feature_df["date"] >= datetime.date(datetime.strptime("2006-1-1", "%Y-%m-%d"))
]


data_train = feature_df_train.loc[
    :, ~feature_df.columns.isin(["date", "City", "precipitation"])
]
data_test = feature_df_test.loc[
    :, ~feature_df.columns.isin(["date", "City", "precipitation"])
]

# try out featurewiz
target = "precip_binary"
features, train = featurewiz(
    data_train,
    target,
    corr_limit=0.7,
    verbose=2,
    sep=",",
    header=0,
    test_data=data_test,
    feature_engg="",
    category_encoders="",
)

print(features)
print(train)

# create train test split
# feature_df_train = feature_df[feature_df["date"] < datetime.date(datetime.strptime('2006-1-1', '%Y-%m-%d'))]
# feature_df_test = feature_df[feature_df["date"] >= datetime.date(datetime.strptime('2006-1-1', '%Y-%m-%d'))]
#
# # now get x_train, y_train, x_test, and y_test variables
# x_train = feature_df_train.loc[:,~feature_df_train.columns.isin(["date", "precip_binary", "City", "precipitation"])]
# cols = list(x_train.columns)
# y_train = feature_df_train["precip_binary"]
# x_test = feature_df_test.loc[:,~feature_df_test.columns.isin(["date", "precip_binary","City", "precipitation"])]
# y_test = feature_df_test["precip_binary"]
#
# # normalize all data
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
#
#
#
# # use SMOTE to fix class imbalance
# oversample = SMOTE()
# x_train, y_train = oversample.fit_resample(x_train, y_train)

# print(y_train.value_counts())

# clf = RandomForestClassifier(verbose=100, n_jobs=-1)
# clf.fit(x_train, y_train)
# # print(np.mean(get_accuracy(result['test_pred_dry_actual_dry'], result['test_pred_wet_actual_wet'],
# #                            result['test_pred_wet_actual_dry'], result['test_pred_dry_actual_wet'])))
# acc, cm = test_score(clf, y_test, scaler.fit_transform(x_test))
# print(acc)
#
# importances = list(clf.feature_importances_)
#
# feature_importances_df = pd.DataFrame({"feature": cols, "importance": importances}).sort_values(by="importance",
#                                                                                                    ascending=False)
# print(feature_importances_df.head(20).to_string())


# want to keep some well performing features regardless of correlations, so extract these features to keep
# keep_features = list(feature_importances_df["feature"][:10])
# print("Features to keep: ", keep_features)


# now fit a model after dropping highly correlated features
# corrs = pickle_load("/Volumes/Transcend/DownscalingData/dataModels/correlationTable")
#
# corrs = corrs.dropna()
# highly_correlated = corrs[abs(corrs["pearson coefficient"]) > 0.6]
#
# drop = []
# for i in highly_correlated["predictor2"]:
#     if i in keep_features:
#         continue
#     else:
#         drop.append(i)
#
#
# df_features_removed = df_day.drop(columns=highly_correlated["predictor2"].values)
#
# #result = cont_cont_correlation(df_features_removed, df_features_removed.columns[6:], "precipitation")
#
# #df_features_removed.drop(columns =
# ["Latitude",'Longitude','lat1', 'lon1', 'lat2', 'lon2', 'lat3','lon3', 'lat4', 'lon4'], inplace=True)
#
#
# #pred contiains only features and date
# pred = []
# for i in df_features_removed.columns:
#     if i == "City" or i =="precipitation":
#         continue
#     else:
#         pred.append(i)
#
#
# feature_df_opt = df_features_removed.loc[:,df_features_removed.columns.isin(pred)]
#
# #print(feature_df_opt.head().to_string())
# #make a train test split
# feature_df_opt_train =
# feature_df_opt[feature_df_opt["date"] < datetime.date(datetime.strptime('2006-1-1', '%Y-%m-%d'))]
# feature_df_opt_test =
# feature_df_opt[feature_df_opt["date"] >= datetime.date(datetime.strptime('2006-1-1', '%Y-%m-%d'))]
#
# # now get x_train/test and y values
# x_train_opt = feature_df_opt_train.loc[:,~feature_df_opt_train.columns.isin(["date", "precip_binary"])]
# y_train_opt = feature_df_opt_train["precip_binary"]
# x_test_opt = feature_df_opt_test.loc[:,~feature_df_opt_test.columns.isin(["date", "precip_binary"])]
# y_test_opt = feature_df_opt_test["precip_binary"]
#
# # perform normalization using min max scaler
# scale = MinMaxScaler()
# x_train_opt_scaled = scale.fit_transform(x_train_opt)
#
# # use smote to balance dataset
# over = SMOTE()
# x_train_opt_scaled, y_train_opt = over.fit_resample(x_train_opt_scaled, y_train_opt)
# #print(y_train_opt.value_counts())
#
# #create a logistic regression and random forest model
#
# lr = LogisticRegression(n_jobs=-1)
# rf = RandomForestClassifier(verbose=100, n_jobs=-1)
# rf2 = RandomForestClassifier(verbose=100, n_jobs=-1)
# #svc = SVC()
# #knn = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
#
#
# print("FEATURES to TRAIN: ", x_train_opt.columns)
# lr_result = fit_model(lr, x_train_opt_scaled, y_train_opt)
# print("Logistic regression cv result: ",
#       np.mean(get_accuracy(lr_result['test_pred_dry_actual_dry'], lr_result['test_pred_wet_actual_wet'],
#                            lr_result['test_pred_wet_actual_dry'], lr_result['test_pred_dry_actual_wet'])) )
#
# lr.fit(x_train_opt_scaled, y_train_opt)
# # get test accuracy
# print("LOGIT TEST RESULTS:")
# accuracy, cm = test_score(lr, y_test_opt, scale.fit_transform(x_test_opt))
# print("lr Test Accuracy: ", accuracy)
# print("lr Confusion Matrix: ", cm)
#
#
# #rf_result = fit_model(rf, x_train_opt_scaled, y_train_opt)
#
# #pickle_dump(rf_result, "/Volumes/Transcend/DownscalingData/dataModels/rf_byDay_keepSome")
# rf_result = pickle_load("/Volumes/Transcend/DownscalingData/dataModels/rf_byDay_dropAll")
# # print("Random forest cv result: ",
# #      np.mean(get_accuracy(rf_result['test_pred_dry_actual_dry'], rf_result['test_pred_wet_actual_wet'],
# #                           rf_result['test_pred_wet_actual_dry'], rf_result['test_pred_dry_actual_wet'])) )
#
#
# rf.fit(x_train_opt_scaled, y_train_opt)
# # get test acc
# print('RANDOM FOREST TEST RESULTS:')
# acc, cm2 = test_score(rf, y_test_opt, scale.fit_transform(x_test_opt))
# print("Random Forest Test Accuracy: ", acc)
# print("Random Forest Confusion Matrix: ", cm2)
# print("Test Precision: ", get_precision(res["test_pred_wet_actual_wet"], res["test_pred_wet_actual_dry"]))
# print("Test Recall: ", get_recall(res["test_pred_wet_actual_wet"], res["test_pred_dry_actual_wet"]))
# svc_result = fit_model(svc,x_train_opt_scaled, y_train_opt )
# knn_result = fit_model(knn, x_train_opt_scaled, y_train_opt)
# print("knn cv result: ",
#      np.mean(get_accuracy(knn_result['test_pred_dry_actual_dry'],knn_result['test_pred_wet_actual_wet'],
#                           knn_result['test_pred_wet_actual_dry'], knn_result['test_pred_dry_actual_wet'])) )
# knn.fit(x_train_opt_scaled, y_train_opt)
# # get test acc
# knnAcc, knnCm = test_score(knn,y_test_opt, scale.fit_transform(x_test_opt))
# print("KNN Test Accuracy: ", knnAcc)
# print("KNN Confusion Matrix: ", knnCm)


# use random forest feature selection and try it with other models


#
