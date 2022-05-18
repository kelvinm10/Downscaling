import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from featurewiz import featurewiz
from preprocessing import pickle_dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR


def print_significant_summary(df_rain, y):
    x_significant = df_rain.iloc[
        :,
        [
            1,
            2,
            3,
            4,
            7,
            9,
            10,
            11,
            12,
            14,
            15,
            16,
            18,
            20,
            21,
            24,
            25,
            26,
            27,
            29,
            30,
            31,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            41,
            42,
            43,
            47,
            49,
            52,
            54,
            56,
            57,
            58,
        ],
    ]
    est_significant = sm.OLS(y, x_significant.values)
    est2 = est_significant.fit()
    print(est2.summary())


def outlier_treatment(datacolumn):
    sorted(datacolumn)
    Q1, Q3 = np.percentile(datacolumn, [25, 75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range, upper_range


def support_vector_regression(x_train, y_train):
    # params = {"kernel":["linear", "poly", "rbf", "sigmoid"],
    # "degree":[1,2,3],
    # "gamma":["scale", "auto"],
    # "C":[0.01, 0.1,1,10,100]
    #  }

    model = SVR()
    # search_model = RandomizedSearchCV(model, params, random_state=0)
    model.fit(x_train, y_train)
    # print(search_model.best_params_)
    return model


def nn_fit(x_train, y_train):
    param_grid = {
        "hidden_layer_sizes": [(50, 50, 50), (50, 100, 50), (100, 1)],
        "activation": ["relu", "tanh", "logistic"],
        "alpha": [0.0001, 0.05, 0.01, 1],
        "learning_rate": ["constant", "adaptive"],
        "solver": ["adam", "sgd"],
    }

    model = MLPRegressor()
    search_model = RandomizedSearchCV(model, param_grid, random_state=0)
    search_model.fit(x_train, y_train)
    print(search_model.best_params_, search_model.best_score_)
    return search_model.best_estimator_


def rf_fit(x_train, y_train):
    param_grid = {
        "n_estimators": [100, 150, 200],
        "criterion": ["squared_error", "absolute_zero", "poisson"],
        "bootstrap": [True, False],
    }
    model = RandomForestRegressor(n_jobs=-1)
    search_model = RandomizedSearchCV(model, param_grid, random_state=0)
    search_model.fit(x_train, y_train)
    print(search_model.best_params_, search_model.best_score_)
    return search_model.best_estimator_


def main():
    df = pd.read_pickle("/Volumes/Transcend/DownscalingData/dataModels/df_day")
    df_rain = df[df["precipitation"] > 0].reset_index(drop=True)

    print(df["precip_binary"].value_counts() / len(df))
    df_rain = df_rain.drop(["precip_binary", "City"], axis=1)
    print(df_rain.shape)
    # px.histogram(df_rain, "precipitation").show()

    # get all predictor values in x and precipitation value in y
    x = df_rain.loc[:, ~df_rain.columns.isin(["date", "precipitation"])]
    y = df_rain["precipitation"]

    x2 = sm.add_constant(x.values)
    est = sm.OLS(y, x2)
    est2 = est.fit()
    # print(est2.summary())
    # significant predictors: x1 - x4, x7, x9-x12, x14 - x16, x18, x20-x21, x24 - x27, x29-x31, x33-x39, x41-x43,
    # x47, x49, x52, x54, x56-x58

    # drop predictors x5-x6, x8, x13, x17, x19, x22-x23, x28, x32, x40, x44-x46, x48, x50-x51, x53, x55, x59-x60
    # also need to take into account highly correlated variables
    # print_significant_summary(df_rain, y)

    # train test split (first 80% in training data, latest 20% test
    train = df_rain[0 : round(0.8 * len(df_rain))]
    test = df_rain[round(0.8 * len(df_rain)) :]
    # drop date column
    train = train.drop(["date"], axis=1)
    test = test.drop(["date"], axis=1)

    # features, idk = featurewiz(train, "precipitation")
    # print(features)
    # print(idk)

    features = [
        "rsdsdiff4",
        "uas2",
        "rlus3",
        "hfss3",
        "rsdsdiff2",
        "hfls3",
        "uas1",
        "clt3",
        "pr1",
        "vas4",
        "pr3",
        "pr4",
        "rsds2",
        "clt1",
        "precipitation",
    ]
    train = train[features]
    test = test[features]

    # feats, train = featurewiz(train, "precipitation", feature_engg=["interactions", "groupby"])

    x_train = train.loc[:, ~train.columns.isin(["precipitation"])]

    # scale features with both standard scaler and minmax
    minMax = MinMaxScaler()
    std = StandardScaler()
    minMax.fit(x_train.values)
    std.fit(x_train.values)
    x_train_min_max = minMax.transform(x_train.values)
    x_train_std = std.transform(x_train.values)

    y_train = train["precipitation"]
    x2_minMax = sm.add_constant(x_train_min_max)
    x2_std = sm.add_constant(x_train_std)
    est_min_max = sm.OLS(y_train, x2_minMax)
    est_std = sm.OLS(y_train, x2_std)
    est_fitted_minMax = est_min_max.fit()
    est_fitted_std = est_std.fit()
    print(est_fitted_minMax.summary())
    print(est_fitted_std.summary())

    # get x_test and y_test and predict using fitted ols models BASE MODEL
    x_test = test.drop(["precipitation"], axis=1)
    y_test = test["precipitation"]

    x_test_min_max = minMax.transform(x_test)
    x_test_std = minMax.transform(x_test)

    # =============== LINEAR REGRESSION ====================

    # yhat_minMax = est_fitted_minMax.predict(sm.add_constant(x_test_min_max))
    # yhat_std = est_fitted_std.predict(sm.add_constant(x_test_std))
    #
    # print("Min Max Root MSE: ", mean_squared_error(y_test, yhat_minMax))
    # print("Standard scaler MSE: ", mean_squared_error(y_test, yhat_std))
    #
    # print("Min Max Root MAE: ", mean_absolute_error(y_test, yhat_minMax))
    # print("Standard scaler MAE: ", mean_absolute_error(y_test, yhat_std))

    # linear regression residuals
    # px.scatter(x=np.linspace(0,28546, 28546), y=est_fitted_minMax.resid, title="Linear Regression Residuals").show()
    # px.scatter(x=np.linspace(0,28546,28546), y=est_fitted_std.resid).show()
    # px.histogram(df_rain, "precipitation").show()
    # px.box(df_rain, "precipitation").show()

    # ==================== SUPPORT VECTOR REGRESSION ========================

    # svr = support_vector_regression(x_train_min_max, y_train)
    # yhat_svr = svr.predict(x_test_min_max)
    #
    # print()
    # print("SVR MSE: ", mean_squared_error(y_test, yhat_svr))
    # print("SVR MAE: ", mean_absolute_error(y_test, yhat_svr))

    # ==================== NEURAL NETWORK ==========================

    # print("NN")
    # nn = nn_fit(x_train_min_max, y_train)
    # nn_yhat = nn.predict(x_test_min_max)
    #
    # print()
    # print("NN MSE: ", mean_squared_error(y_test, nn_yhat))
    # print("NN MAE: ", mean_absolute_error(y_test, nn_yhat))

    #  ===================== RANDOM FOREST ==========================
    # rf = RandomForestRegressor(n_jobs=-1, n_estimators=150, criterion="squared_error", bootstrap=True, random_state=0)
    # rf.fit(x_train_min_max, y_train)
    # rf_yhat = rf.predict(x_test_min_max)
    #
    # print()
    # print("RF MSE: ", mean_squared_error(y_test, rf_yhat))
    # print("RF MAE: ", mean_absolute_error(y_test, rf_yhat))

    # residuals = y_test - rf_yhat
    #
    # px.scatter(residuals, title="Random Forest Residuals").show()
    # px.scatter(y_test-yhat_svr, title="Support Vector Regression Residuals").show()
    # px.scatter(y_test-nn_yhat, title="Neural Network Residuals").show()

    # ===================== MODELING BY SEASON ========================

    # divide rain dataframe into 2 seasons: wet season and dry season (Nov - April & May - October)

    df_rain["date"] = pd.to_datetime(df_rain["date"])
    wet_season_df = df_rain[
        (df_rain["date"].dt.month > 10) | (df_rain["date"].dt.month < 5)
    ].reset_index(drop=True)
    dry_season_df = df_rain[
        (df_rain["date"].dt.month > 4) & (df_rain["date"].dt.month < 11)
    ].reset_index(drop=True)

    print("wet season df: ", wet_season_df.shape)
    print("dry season df: ", dry_season_df.shape)

    print(wet_season_df.head(10))
    print(dry_season_df.head(10))

    # split into training and testing sets, and use cross validation for parameter tuning, first filter to
    # use same features as above

    train_wet = wet_season_df[0 : round(0.8 * len(wet_season_df))]
    test_wet = wet_season_df[round(0.8 * len(wet_season_df)) :]

    train_dry = dry_season_df[0 : round(0.8 * len(dry_season_df))]
    test_dry = dry_season_df[round(0.8 * len(dry_season_df)) :]

    train_wet = train_wet.drop(["date"], axis=1)
    train_dry = train_dry.drop(["date"], axis=1)

    all_wet = wet_season_df.drop(["date"], axis=1)
    all_dry = dry_season_df.drop(["date"], axis=1)

    # features,_ = featurewiz(train_wet, "precipitation")
    # features2,_ = featurewiz(train_dry, "precipitation")
    # print("wet season features: ", features)
    # print("dry season features: ", features2)

    wet_features = [
        "pr2",
        "hfss3",
        "rsdsdiff4",
        "hfss1",
        "rlus3",
        "huss4",
        "uas1",
        "pr1",
        "clt2",
        "uas3",
        "rsus1",
        "pr4",
        "clt4",
        "precipitation",
    ]
    dry_features = [
        "prc2",
        "ps2",
        "rsdsdiff3",
        "hfls3",
        "clt3",
        "rsdsdiff4",
        "clt1",
        "rlus2",
        "vas2",
        "prc3",
        "rsus4",
        "vas3",
        "rlds2",
        "pr1",
        "hfls4",
        "precipitation",
    ]

    all_wet = all_wet[wet_features]
    all_dry = all_dry[dry_features]

    # now get only best features extracted from featurewiz
    train_wet = train_wet[wet_features]
    test_wet = test_wet[wet_features]

    train_dry = train_dry[dry_features]
    test_dry = test_dry[dry_features]

    x_train_wet = train_wet.drop(["precipitation"], axis=1)
    y_train_wet = train_wet["precipitation"]

    x_test_wet = test_wet.drop(["precipitation"], axis=1)
    y_test_wet = test_wet["precipitation"]

    x_train_dry = train_dry.drop(["precipitation"], axis=1)
    y_train_dry = train_dry["precipitation"]

    x_test_dry = test_dry.drop(["precipitation"], axis=1)
    y_test_dry = test_dry["precipitation"]

    all_wet_x = all_wet.drop(["precipitation"], axis=1)
    all_wet_y = all_wet["precipitation"]

    all_dry_x = all_dry.drop(["precipitation"], axis=1)
    all_dry_y = all_dry["precipitation"]
    # train and test datasets are set, now scale data using min max scaler

    mm_scaler = MinMaxScaler()
    mm_scaler_2 = MinMaxScaler()
    # mm_scaler.fit(x_train_wet)
    # x_train_wet_scaled = mm_scaler.transform(x_train_wet)
    # x_test_wet_scaled = mm_scaler.transform(x_test_wet)
    #
    # mm_scaler.fit(x_train_dry)
    # x_train_dry_scaled = mm_scaler.transform(x_train_dry)
    # x_test_dry_scaled = mm_scaler.transform(x_test_dry)
    mm_scaler.fit(all_wet_x)
    all_wet_x = mm_scaler.transform(all_wet_x)

    mm_scaler_2.fit(all_dry_x)
    all_dry_x = mm_scaler_2.transform(all_dry_x)

    wet_model = rf_fit(all_wet_x, all_wet_y)
    dry_model = rf_fit(all_dry_x, all_dry_y)
    pickle_dump(
        mm_scaler,
        "/Volumes/Transcend/DownscalingData/FutureDataSaves/minmax_scaler_regression_wet.pkl",
    )
    pickle_dump(
        mm_scaler_2,
        "/Volumes/Transcend/DownscalingData/FutureDataSaves/minmax_scaler_regression_dry.pkl",
    )
    pickle_dump(
        wet_model,
        "/Volumes/Transcend/DownscalingData/FutureDataSaves/wet_random_forest_regression.pkl",
    )
    pickle_dump(
        dry_model,
        "/Volumes/Transcend/DownscalingData/FutureDataSaves/dry_random_forest_regression.pkl",
    )

    # ============================= WET SEASON MODELING =============================
    # lr_model = LinearRegression()
    #
    # #linear regression
    # lr_model.fit(x_train_wet_scaled, y_train_wet)
    #
    #
    # # Support vector regression
    # print("svr")
    # svr = support_vector_regression(x_train_wet_scaled, y_train_wet)
    # print()
    # # Neural Network
    # print("NN")
    # nn = nn_fit(x_train_wet_scaled, y_train_wet)
    # print()
    # # Random Forest
    # print("RF")
    # rf = rf_fit(x_train_wet_scaled, y_train_wet)
    #
    #
    #
    # lr_yhat = lr_model.predict(x_test_wet_scaled)
    # svr_yhat = svr.predict(x_test_wet_scaled)
    # nn_yhat = nn.predict(x_test_wet_scaled)
    # rf_yhat = rf.predict(x_test_wet_scaled)
    #
    # print("WET SEASON LINEAR REGRESSION MSE: ", mean_squared_error(lr_yhat, y_test_wet))
    # print("WET SEASON LINEAR REGRESSION MAE: ", mean_absolute_error(lr_yhat, y_test_wet))
    # print("WET SEASON SVR MSE: ", mean_squared_error(svr_yhat, y_test_wet))
    # print("WET SEASON SVR MAE: ", mean_absolute_error(svr_yhat, y_test_wet))
    # print("WET SEASON NN MSE: ", mean_squared_error(nn_yhat, y_test_wet))
    # print("WET SEASON NN MAE: ", mean_absolute_error(nn_yhat, y_test_wet))
    # print("WET SEASON RF MSE: ", mean_squared_error(rf_yhat, y_test_wet))
    # print("WET SEASON RF MAE: ", mean_absolute_error(rf_yhat, y_test_wet))
    #
    # px.scatter(y_test_wet - lr_yhat, title="Wet Season Linear Regression Residuals").show()
    # px.scatter(y_test_wet - svr_yhat, title="Wet Season SVR Residuals").show()
    # px.scatter(y_test_wet - nn_yhat, title="Wet Season NN Residuals").show()
    # px.scatter(y_test_wet - rf_yhat, title="Wet Season RF Residuals").show()
    #
    # px.histogram(y_test_wet - lr_yhat, title=" Wet Season Linear Regression Residual Distribution").show()
    # px.histogram(y_test_wet - svr_yhat, title="Wet Season SVR Residual Distribution").show()
    # px.histogram(y_test_wet - nn_yhat, title="Wet Season NN Residual Distribution").show()
    # px.histogram(y_test_wet - rf_yhat, title="Wet Season RF Residual Distribution").show()
    #
    # # ============================= DRY SEASON MODELING =============================
    # lr_model = LinearRegression()
    #
    # # linear regression
    # lr_model.fit(x_train_dry_scaled, y_train_dry)
    #
    # # Support vector regression
    # print("svr")
    # svr = support_vector_regression(x_train_dry_scaled, y_train_dry)
    # print()
    # # Neural Network
    # print("NN")
    # nn = nn_fit(x_train_dry_scaled, y_train_dry)
    # print()
    # # Random Forest
    # print("RF")
    # rf = rf_fit(x_train_dry_scaled, y_train_dry)
    #
    # lr_yhat = lr_model.predict(x_test_dry_scaled)
    # svr_yhat = svr.predict(x_test_dry_scaled)
    # nn_yhat = nn.predict(x_test_dry_scaled)
    # rf_yhat = rf.predict(x_test_dry_scaled)
    #
    # print("dry SEASON LINEAR REGRESSION MSE: ", mean_squared_error(lr_yhat, y_test_dry))
    # print("dry SEASON LINEAR REGRESSION MAE: ", mean_absolute_error(lr_yhat, y_test_dry))
    # print("dry SEASON SVR MSE: ", mean_squared_error(svr_yhat, y_test_dry))
    # print("dry SEASON SVR MAE: ", mean_absolute_error(svr_yhat, y_test_dry))
    # print("dry SEASON NN MSE: ", mean_squared_error(nn_yhat, y_test_dry))
    # print("dry SEASON NN MAE: ", mean_absolute_error(nn_yhat, y_test_dry))
    # print("dry SEASON RF MSE: ", mean_squared_error(rf_yhat, y_test_dry))
    # print("dry SEASON RF MAE: ", mean_absolute_error(rf_yhat, y_test_dry))
    #
    #
    #
    #
    #
    #
    # px.scatter(y_test_dry-lr_yhat, title=" dry Season Linear Regression Residuals").show()
    # px.scatter(y_test_dry - svr_yhat, title=" dry Season SVR Residuals").show()
    # px.scatter(y_test_dry - nn_yhat, title=" dry Season NN Residuals").show()
    # px.scatter(y_test_dry - rf_yhat, title=" dry Season RF Residuals").show()
    #
    # px.histogram(y_test_dry-lr_yhat, title=" Dry Season Linear Regression Residual Distribution").show()
    # px.histogram(y_test_dry - svr_yhat, title=" Dry Season SVR Residual Distribution").show()
    # px.histogram(y_test_dry - nn_yhat, title=" Dry Season NN Residual Distribution").show()
    # px.histogram(y_test_dry - rf_yhat, title=" Dry Season RF Residual Distribution").show()


if __name__ == "__main__":
    main()
