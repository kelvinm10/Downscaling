import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from featurewiz import featurewiz
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
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

    yhat_minMax = est_fitted_minMax.predict(sm.add_constant(x_test_min_max))
    yhat_std = est_fitted_std.predict(sm.add_constant(x_test_std))

    print("Min Max Root MSE: ", mean_squared_error(y_test, yhat_minMax))
    print("Standard scaler MSE: ", mean_squared_error(y_test, yhat_std))

    print("Min Max Root MAE: ", mean_absolute_error(y_test, yhat_minMax))
    print("Standard scaler MAE: ", mean_absolute_error(y_test, yhat_std))

    # linear regression residuals
    # px.scatter(x=np.linspace(0,28546, 28546), y=est_fitted_minMax.resid).show()
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


if __name__ == "__main__":
    main()
