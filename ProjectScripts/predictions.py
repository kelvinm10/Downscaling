import datetime
import os
from functools import reduce

import cftime
import numpy as np
import pandas as pd
import xarray as xr
import xarray.coding.times
from preprocessing import format_local_dfs, pickle_dump, pickle_load


# Function to load and transform the nc file to a pandas dataframe
# does this by filtering for only the san diego region
# (Lat: 31-33, lon: 61.25-63.75) then returning the filtered dataset
# as a pandas dataframe
def load_dataframe(path, lat_bnds=[31.25, 34], lon_bnds=[62, 65]):
    ds = xr.open_dataset(path, decode_cf=True, decode_times=True)
    subset = ds.sel(lat=slice(*lat_bnds), lon=slice(*lon_bnds))
    return subset.to_dataframe()


# this function will load all of the dataframes in alphabetical order.
# PARAMETER: directory -> (string) directory to find all of the downloaded data
# RETURNS: dataHolder -> (list) of length 60 (15 predictors, 4 files for each predictor). list of pandas dataframes
def load_all_df(directory):
    fileHolder = []
    for filename in os.listdir(directory):
        split = filename.split("._")
        if len(split) == 2:
            filename = split[1]
        f = os.path.join(directory, filename)
        fileHolder.append(f)

    result = sorted(set(fileHolder))
    dataHolder = []
    for i in result:
        print(i)
        if (
            i
            == "/Volumes/Transcend/DownscalingData/FutureData/vas_day_UKESM1-0-LL_ssp126_r8i1p1f2_gn_20150101-20491230.nc"
        ):
            curr_df = load_dataframe(i, lat_bnds=[31.875, 34])
        elif (
            i
            == "/Volumes/Transcend/DownscalingData/FutureData/uas_day_UKESM1-0-LL_ssp370_r8i1p1f2_gn_20150101-20491230.nc"
        ):
            curr_df = load_dataframe(i, lon_bnds=[61, 64])
        else:
            curr_df = load_dataframe(i)
        curr_df.reset_index(inplace=True)
        curr_df = curr_df[curr_df["bnds"] == 0].reset_index(drop=True)
        print(curr_df.head().to_string())
        print(curr_df.shape)
        dataHolder.append(curr_df)

    return dataHolder


def merge_global_data(global_data):
    list_of_dfs = []
    for i in global_data:

        list_of_dfs.append(i)

    # print(list_of_dfs)
    # df_final = reduce(
    #     lambda left, right: pd.merge_asof(left, right, on="lat", by=["time", "lon"]),
    #     list_of_dfs,
    # )
    df_final = reduce(
        lambda left, right: pd.merge(left, right, on=["time", "lat", "lon"]),
        list_of_dfs,
    )

    return df_final


# function to find the closest lat lon pairs to match up the local data with a specific gauge in the global model
def find_closest_coord(
    lat, lon, target_lat=[31.875, 33.125], target_lon=[62.8125, 64.6875], index=0
):

    arr = np.asarray(target_lat)
    arr2 = np.asarray(target_lon)
    if index == 0:
        i = (np.abs(arr - lat)).argmin()
        i2 = (np.abs(arr2 - lon)).argmin()
    elif index == 1:
        i = (np.abs(arr - lat)).argmax()
        i2 = (np.abs(arr2 - lon)).argmin()
    elif index == 2:
        i = (np.abs(arr - lat)).argmin()
        i2 = (np.abs(arr2 - lon)).argmax()
    else:
        i = (np.abs(arr - lat)).argmax()
        i2 = (np.abs(arr2 - lon)).argmax()

    closest_lat = arr[i]
    closest_lon = arr2[i2]

    return [closest_lat, closest_lon]


def classification_prediction(df, model, scaler, features):
    df = df[features]
    df = scaler.transform(df)
    yhat = model.predict(df)
    return yhat


def regression_prediction(df, model, scaler, features):
    df = df[features]
    df = scaler.transform(df)
    yhat = model.predict(df)
    return yhat


def main():

    lat_lon_dict = {
        "AguaCaliente": [32.96, 63.7],
        "Barona": [33, 63.16],
        "BarrettLake": [32.68, 63.33],
        "Bonita": [32.66, 62.97],
        "BorregoCRS": [33.22, 63.66],
        "BorregoPalm": [33.27, 63.59],
        "Carlsbad": [33.13, 62.72],
        "CoyoteCreek": [33.37, 63.58],
        "Descanso": [32.85, 63.38],
        "DulzuraSummit": [32.62, 63.26],
        "LaJollaAmago": [33.28, 63.14],
        "LakeCuyamaca": [32.99, 63.41],
        "LakeHenshaw": [33.24, 63.24],
        "LakeMurray": [32.78, 62.96],
        "LakeHodges": [33.07, 62.89],
        "LakeWohlford": [33.17, 63],
        "LaMesa": [32.77, 62.98],
        "LosCoches": [32.84, 63.1],
        "MiramarLake": [32.91, 62.9],
        "MorenaLake": [32.68, 63.46],
        "MtLagunaCRS": [32.86, 63.58],
        "MtWoodson": [33.01, 63.04],
        "OcotilloWells": [33.15, 63.82],
        "Oceanside": [33.21, 62.65],
        "PineHills": [33.02, 63.36],
        "PalomarObservatory": [33.36, 63.14],
        "RamonaCRS": [33.05, 63.14],
        "Poway": [32.95, 62.95],
        "Ranchita": [33.21, 63.47],
        "RanchoBernardo": [33.02, 62.92],
        "RinconSprings": [33.29, 63.04],
        "SanFelipe": [33.1, 63.53],
        "SandiaCreekRd": [33.41, 62.76],
        "SanmarcosLandfill": [33.09, 62.81],
        "SanOnofre": [33.35, 62.47],
        "SantaYsabel": [33.11, 63.33],
        "Santee": [32.84, 62.98],
        "SanYsidro": [32.55, 62.93],
        "Sutherland": [33.12, 63.21],
        "TierradelSol": [32.65, 63.68],
        "WitchCreek": [33.07, 63.26],
        "ValleyCenter": [33.22, 62.96],
    }

    path = "/Volumes/Transcend/DownscalingData/FutureData/"

    save_path = "/Volumes/Transcend/DownscalingData/FutureDataSaves/"

    # dataHolder = load_all_df(path)
    # pickle_dump(dataHolder, save_path + "dataHolder.pkl")
    dataHolder = pickle_load(save_path + "dataHolder.pkl")

    city_df = pd.DataFrame.from_dict(
        lat_lon_dict, orient="index", columns=["lat", "lon"]
    )
    city_df.index.set_names(["city"], inplace=True)
    city_df.reset_index(inplace=True)
    # print(city_df)
    filtered_data = []
    for i in dataHolder:
        if "height" in i.columns:
            i.drop(columns=["height"], inplace=True)
        i.drop(["bnds", "lat_bnds", "lon_bnds", "time_bnds"], axis=1, inplace=True)

        if len(i) == 7300:
            i.time = i.time.astype(str)
            filtered_data.append(i)
            print(len(i))
            continue

        i = i[i["time"] <= cftime.Datetime360Day(2035, 1, 1)]
        i.time = i.time.astype(str)
        filtered_data.append(i)
        print(len(i))

    # normal values: lat = [31.875, 33.125]
    #                lon = [62.8125, 64.6875]

    # need to preserve atcual lat and lon values, but create matching lat and lon values for merge
    filtered_data[-1] = filtered_data[-1].rename(columns={"lat": "vas_lat"})
    filtered_data[-2] = filtered_data[-2].rename(columns={"lon": "uas_lon"})
    filtered_data[-4] = filtered_data[-4].rename(
        columns={"lat": "rsdsdiff_lat", "lon": "rsdsdiff_lon"}
    )

    filtered_data[-4] = pd.merge(
        filtered_data[0][["time"]], filtered_data[-4], on="time", how="left"
    )
    print("merging just with time")

    print("New Length: ", len(filtered_data[-4]))
    print("NEW DATA: ", filtered_data[-4])

    filtered_data[-1]["lat"] = filtered_data[0]["lat"].values
    filtered_data[-2]["lon"] = filtered_data[0]["lon"].values
    filtered_data[-4]["lat"] = filtered_data[0]["lat"].values
    filtered_data[-4]["lon"] = filtered_data[0]["lon"].values

    res = merge_global_data(filtered_data)

    print(res)

    cities = list(lat_lon_dict.keys())
    print(cities)

    first_closest_lat = []
    first_closest_lon = []
    second_closest_lat = []
    second_closest_lon = []
    third_closest_lat = []
    third_closest_lon = []
    fourth_closest_lat = []
    fourth_closest_lon = []
    for i in cities:
        closest = find_closest_coord(lat_lon_dict[i][0], lat_lon_dict[i][1])
        second = find_closest_coord(lat_lon_dict[i][0], lat_lon_dict[i][1], index=1)
        third = find_closest_coord(lat_lon_dict[i][0], lat_lon_dict[i][1], index=2)
        fourth = find_closest_coord(lat_lon_dict[i][0], lat_lon_dict[i][1], index=3)
        print(i, "closest coordinate: " + str(closest))
        print(i, "2nd closest coordinate: " + str(second))
        print(i, "3rd closest coordinate: " + str(third))
        print(i, "4th closest coordinate: " + str(fourth))

        first_closest_lat.append(closest[0])
        first_closest_lon.append(closest[1])
        second_closest_lat.append(second[0])
        second_closest_lon.append(second[1])
        third_closest_lat.append(third[0])
        third_closest_lon.append(third[1])
        fourth_closest_lat.append(fourth[0])
        fourth_closest_lon.append(fourth[1])

    city_df = pd.DataFrame(
        {
            "city": cities,
            "lat1": first_closest_lat,
            "lon1": first_closest_lon,
            "lat2": second_closest_lat,
            "lon2": second_closest_lon,
            "lat3": third_closest_lat,
            "lon3": third_closest_lon,
            "lat4": fourth_closest_lat,
            "lon4": fourth_closest_lon,
        }
    )

    closest_merge = pd.merge(
        res,
        city_df[["city", "lat1", "lon1"]],
        left_on=["lat", "lon"],
        right_on=["lat1", "lon1"],
    )
    second_merge = pd.merge(
        res,
        city_df[["city", "lat2", "lon2"]],
        left_on=["lat", "lon"],
        right_on=["lat2", "lon2"],
    )
    third_merge = pd.merge(
        res,
        city_df[["city", "lat3", "lon3"]],
        left_on=["lat", "lon"],
        right_on=["lat3", "lon3"],
    )
    fourth_merge = pd.merge(
        res,
        city_df[["city", "lat4", "lon4"]],
        left_on=["lat", "lon"],
        right_on=["lat4", "lon4"],
    )

    test = pd.merge(
        closest_merge, second_merge, on=["time", "city"], suffixes=("1", "2")
    )
    next = pd.merge(test, third_merge, on=["time", "city"])

    final_df = pd.merge(next, fourth_merge, on=["time", "city"], suffixes=("3", "4"))
    drop = []
    for i in list(final_df.columns):
        if "lat" in i:
            drop.append(i)
        elif "lon" in i:
            drop.append(i)

    final_df = final_df.drop(drop, axis=1)

    final_df = final_df.sort_values(by=["city", "time"])

    cols = ["time", "city"] + [
        col for col in final_df.columns if col != "time" and col != "city"
    ]

    final_df = final_df[cols].reset_index(drop=True)

    # change time column back to datetime 360 day object
    dates = []
    months = []
    for i in final_df["time"]:
        strp = i.split(" ")
        vals = strp[0].split("-")
        dates.append(cftime.Datetime360Day(int(vals[0]), int(vals[1]), int(vals[2])))
        months.append(int(vals[1]))

    final_df["time"] = dates
    final_df["month"] = months

    # get only future data from 2023 to 2034
    predict_df = final_df[final_df["time"] >= cftime.Datetime360Day(2023, 1, 1)]
    print(predict_df.shape)
    predict_df = predict_df.dropna()
    print(predict_df.shape)
    print(predict_df.isna().sum())

    # now ready to predict using our models, first classification
    clf = pickle_load(
        "/Volumes/Transcend/DownscalingData/FutureDataSaves/random_forest.pkl"
    )
    scaler = pickle_load(
        "/Volumes/Transcend/DownscalingData/FutureDataSaves/minmaxclf.pkl"
    )

    wet_reg_model = pickle_load(
        "/Volumes/Transcend/DownscalingData/FutureDataSaves/wet_random_forest_regression.pkl"
    )
    dry_reg_model = pickle_load(
        "/Volumes/Transcend/DownscalingData/FutureDataSaves/dry_random_forest_regression.pkl"
    )

    wet_scaler = pickle_load(
        "/Volumes/Transcend/DownscalingData/FutureDataSaves/minmax_scaler_regression_wet.pkl"
    )
    dry_scaler = pickle_load(
        "/Volumes/Transcend/DownscalingData/FutureDataSaves/minmax_scaler_regression_dry.pkl"
    )

    clf_features = [
        "clt4",
        "hfls1",
        "rsdsdiff1",
        "hfls2",
        "rsdsdiff2",
        "hfls3",
        "rsdsdiff3",
        "hfls4",
        "rsdsdiff4",
        "ps2",
        "ps4",
        "uas2",
        "vas4",
        "uas1",
        "hfss1",
        "hfss2",
        "rsus3",
        "huss4",
        "rsus4",
        "pr1",
        "prc4",
        "pr2",
        "pr3",
        "clt2",
    ]

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
    ]

    classification_predictions = classification_prediction(
        predict_df, clf, scaler, clf_features
    )

    # now, we need to get all the predictions that were = 1 (rain) and pass those into regression

    predict_df["classification"] = classification_predictions
    print(predict_df["classification"].value_counts())

    rain_days = predict_df[predict_df["classification"] == 1]

    wet_rain_days = rain_days[
        (rain_days["month"] > 10) | (rain_days["month"] < 5)
    ].reset_index(drop=True)
    dry_rain_days = rain_days[
        (rain_days["month"] > 4) & (rain_days["month"] < 11)
    ].reset_index(drop=True)

    wet_regression_predictions = regression_prediction(
        wet_rain_days, wet_reg_model, wet_scaler, wet_features
    )
    dry_regression_predictions = regression_prediction(
        dry_rain_days, dry_reg_model, dry_scaler, dry_features
    )
    print(wet_regression_predictions)
    print(dry_regression_predictions)

    wet_rain_days["predicted_precipitation"] = wet_regression_predictions
    dry_rain_days["predicted_precipitation"] = dry_regression_predictions

    # now concatenate both the wet season and dry season dataframes with their predictions
    rain_concat = pd.concat([wet_rain_days, dry_rain_days])

    result_df = predict_df[["time", "city", "classification"]].reset_index(drop=True)
    rain_df = rain_concat[
        ["time", "city", "classification", "predicted_precipitation"]
    ].reset_index(drop=True)

    # Merge entire dataframe with rain dataframe to get regression results into one dataframe
    result_df = pd.merge(
        result_df, rain_df, on=["time", "city", "classification"], how="left"
    )

    # fill NA (rows where classifier predicted no rain) with 0, because a classification of 0 means no rain
    result_df = result_df.fillna(0)

    # save result in csv
    result_df.to_csv(
        "/Volumes/Transcend/DownscalingData/FutureDataSaves/predicted_precipitation_2023-2034.csv"
    )


if __name__ == "__main__":
    main()
