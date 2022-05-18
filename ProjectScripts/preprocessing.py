import os
import pickle
from functools import reduce

import numpy as np
import pandas as pd
import xarray as xr


# Function to load and transform the nc file to a pandas dataframe
# does this by filtering for only the san diego region
# (Lat: 31-33, lon: 61.25-63.75) then returning the filtered dataset
# as a pandas dataframe
def load_dataframe(path):
    ds = xr.open_dataset(path, decode_cf=False)
    lat_bnds, lon_bnds = [31, 33], [61.25, 63.75]
    subset = ds.sel(lat=slice(*lat_bnds), lon=slice(*lon_bnds))
    return subset.to_dataframe()


# dump a new pickle file
def pickle_dump(obj, pathname):
    with open(pathname, "wb") as fp:
        pickle.dump(obj, fp)


# load a pickle file
def pickle_load(pathname):
    with open(pathname, "rb") as fp:
        return pickle.load(fp)


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
        curr_df = load_dataframe(i)
        curr_df.reset_index(inplace=True)
        curr_df = curr_df[curr_df["bnds"] == 0]
        print(curr_df.head().to_string())
        print(curr_df.shape)
        dataHolder.append(curr_df)

    return dataHolder


# this function will concatenate all of the variables from all of the years, and
# sort them accordingly by year.
# PARAMETER: dataList -> list of length 60 containing all of the variables in alphabetical order
# REUTRNS: result -> list of length 15 (15 total predictors), each entry is a pandas dataframe of one predictor
#          Ordered in alphabetical order of predictor's abbreviation
def concat_variables_and_sort(dataList):
    index = 0
    frames = []
    result = []
    for i in range(len(dataList) + 1):
        if index == 0:
            hold = index
            index += 1
        elif index % 4 == 0:
            frames.append(dataList[hold])
            # print("appended ", hold)
            # print("length ", len(frames))
            conc = pd.concat(frames, keys=[1989, 1999, 2005, 2012])
            result.append(conc)
            frames = []
            hold = index
            index += 1
        else:
            frames.append(dataList[index])
            # print("appended ", index)
            index += 1

    return result


# function to create a date column in the global data to match the local data
def create_date_col(df):
    date_col = pd.date_range("1980-1-1 00:00:00", end="2012-12-31 21:00:00", freq="3H")
    # get rid of extra day in leap years, not included in global data
    date_col = date_col[~((date_col.month == 2) & (date_col.day == 29))]
    date_col = date_col.repeat(4)
    df.insert(loc=0, column="Datetime", value=date_col)


# function to find the closest lat lon pairs to match up the local data with a specific gauge in the global model
def find_closest_coord(lat, lon, target_lat=[31, 33], target_lon=[61.25, 63.75]):

    arr = np.asarray(target_lat)
    i = (np.abs(arr - lat)).argmin()
    closest_lat = arr[i]

    arr2 = np.asarray(target_lon)
    i2 = (np.abs(arr2 - lon)).argmin()
    closest_lon = arr2[i2]

    return [closest_lat, closest_lon]


# function to merge all of the global data into one dataframe
def merge_global_data(global_data):
    list_of_dfs = []
    for i in global_data:
        if "height" in i.columns:
            i.drop(columns=["height"], inplace=True)
        i.drop(
            columns=["bnds", "time", "lat_bnds", "lon_bnds", "time_bnds"], inplace=True
        )
        list_of_dfs.append(i)

    df_final = reduce(
        lambda left, right: pd.merge(left, right, on=["Datetime", "lat", "lon"]),
        list_of_dfs,
    )

    return df_final


def format_local_dfs(local_path, lat_lon_dict):
    local_dfs = []
    for i in os.listdir(local_path):
        # ignore tmp files
        if not i.startswith("._"):
            curr_df = pd.read_csv(local_path + i)
            curr_city = i.split("_")[1].split(".")[0]
            curr_df.insert(0, "City", curr_city)
            # find closest coordinates to one of the 4 values in the global data (lat, lon):
            # 1) (31, 61.25)
            # 2) (33, 62.5)
            # 3) (32, 60)
            # 4) (32, 62.5)

            closest_coord = find_closest_coord(
                lat_lon_dict[curr_city][0], lat_lon_dict[curr_city][1]
            )
            # insert original lat lon pairs AND closest to global data pairs
            curr_df.insert(2, "Latitude", lat_lon_dict[curr_city][0])
            curr_df.insert(3, "Longitude", lat_lon_dict[curr_city][1])
            curr_df.insert(4, "lat", closest_coord[0])
            curr_df.insert(5, "lon", closest_coord[1])
            curr_df["Datetime"] = pd.to_datetime(curr_df["Datetime"])
            local_dfs.append(curr_df)

    return local_dfs


def create_final_df(local_df, global_df):
    final_df = pd.merge(local_df, global_df, how="inner", on=["Datetime", "lat", "lon"])

    first_filter = global_df[(global_df["lat"] == 31.0) & (global_df["lon"] == 63.75)]
    second_filter = global_df[(global_df["lat"] == 31.0) & (global_df["lon"] == 66.25)]
    third_filter = global_df[(global_df["lat"] == 33.0) & (global_df["lon"] == 66.25)]

    final_df = pd.merge(final_df, first_filter, how="inner", on=["Datetime"])
    final_df.rename(
        columns={"lat_x": "lat1", "lat_y": "lat2", "lon_x": "lon1", "lon_y": "lon2"},
        inplace=True,
    )
    final_df = pd.merge(final_df, second_filter, how="inner", on=["Datetime"])
    final_df.rename(
        columns={
            "clt_x": "clt1",
            "hfls_x": "hfls1",
            "hfss_x": "hfss1",
            "huss_x": "huss1",
            "pr_x": "pr1",
            "prc_x": "prc1",
            "ps_x": "ps1",
            "rlds_x": "rlds1",
            "rlus_x": "rlus1",
            "rsds_x": "rsds1",
            "rsdsdiff_x": "rsdsdiff1",
            "rsus_x": "rsus1",
            "tas_x": "tas1",
            "uas_x": "uas1",
            "vas_x": "vas1",
        },
        inplace=True,
    )
    final_df.rename(
        columns={
            "clt_y": "clt2",
            "hfls_y": "hfls2",
            "hfss_y": "hfss2",
            "huss_y": "huss2",
            "pr_y": "pr2",
            "prc_y": "prc2",
            "ps_y": "ps2",
            "rlds_y": "rlds2",
            "rlus_y": "rlus2",
            "rsds_y": "rsds2",
            "rsdsdiff_y": "rsdsdiff2",
            "rsus_y": "rsus2",
            "tas_y": "tas2",
            "uas_y": "uas2",
            "vas_y": "vas2",
        },
        inplace=True,
    )
    final_df = pd.merge(final_df, third_filter, how="inner", on=["Datetime"])
    final_df.rename(
        columns={
            "clt_x": "clt3",
            "hfls_x": "hfls3",
            "hfss_x": "hfss3",
            "huss_x": "huss3",
            "pr_x": "pr3",
            "prc_x": "prc3",
            "ps_x": "ps3",
            "rlds_x": "rlds3",
            "rlus_x": "rlus3",
            "rsds_x": "rsds3",
            "rsdsdiff_x": "rsdsdiff3",
            "rsus_x": "rsus3",
            "tas_x": "tas3",
            "uas_x": "uas3",
            "vas_x": "vas3",
        },
        inplace=True,
    )
    final_df.rename(
        columns={
            "clt_y": "clt4",
            "hfls_y": "hfls4",
            "hfss_y": "hfss4",
            "huss_y": "huss4",
            "pr_y": "pr4",
            "prc_y": "prc4",
            "ps_y": "ps4",
            "rlds_y": "rlds4",
            "rlus_y": "rlus4",
            "rsds_y": "rsds4",
            "rsdsdiff_y": "rsdsdiff4",
            "rsus_y": "rsus4",
            "tas_y": "tas4",
            "uas_y": "uas4",
            "vas_y": "vas4",
            "lat_x": "lat3",
            "lat_y": "lat4",
            "lon_x": "lon3",
            "lon_y": "lon4",
        },
        inplace=True,
    )
    return final_df


def main():
    # loop through all data files and load them as a pandas dataframe
    # directory = "/Volumes/Transcend/DownscalingData/ClimateModelData/"
    # dataList_path = "/Volumes/Transcend/DownscalingData/dataModels/dataList"

    # dataHolder = load_all_df(directory)
    # pickle_dump(dataHolder, dataList_path)

    # load saved pickle file with all data
    # dataList = pickle_load(dataList_path)
    #
    # # concatenate all of the corresponding preserving the time series data
    # master_df = concat_variables_and_sort(dataList)
    # # print(master_df[0].head())
    # # Create a date column for all global datasets to merge local data with
    # for i in master_df:
    #     create_date_col(i)
    #
    # # print(master_df[0].head().to_string())
    # local_path = "/Volumes/Transcend/DownscalingData/ObservedData3hr/"
    # lat_lon_dict = {
    #     "AguaCaliente": [32.96, 63.7],
    #     "Barona": [33, 63.16],
    #     "BarrettLake": [32.68, 63.33],
    #     "Bonita": [32.66, 62.97],
    #     "BorregoCRS": [33.22, 63.66],
    #     "BorregoPalm": [33.27, 63.59],
    #     "Carlsbad": [33.13, 62.72],
    #     "CoyoteCreek": [33.37, 63.58],
    #     "Descanso": [32.85, 63.38],
    #     "DulzuraSummit": [32.62, 63.26],
    #     "LaJollaAmago": [33.28, 63.14],
    #     "LakeCuyamaca": [32.99, 63.41],
    #     "LakeHenshaw": [33.24, 63.24],
    #     "LakeMurray": [32.78, 62.96],
    #     "LakeHodges": [33.07, 62.89],
    #     "LakeWohlford": [33.17, 63],
    #     "LaMesa": [32.77, 62.98],
    #     "LosCoches": [32.84, 63.1],
    #     "MiramarLake": [32.91, 62.9],
    #     "MorenaLake": [32.68, 63.46],
    #     "MtLagunaCRS": [32.86, 63.58],
    #     "MtWoodson": [33.01, 63.04],
    #     "OcotilloWells": [33.15, 63.82],
    #     "Oceanside": [33.21, 62.65],
    #     "PineHills": [33.02, 63.36],
    #     "PalomarObservatory": [33.36, 63.14],
    #     "RamonaCRS": [33.05, 63.14],
    #     "Poway": [32.95, 62.95],
    #     "Ranchita": [33.21, 63.47],
    #     "RanchoBernardo": [33.02, 62.92],
    #     "RinconSprings": [33.29, 63.04],
    #     "SanFelipe": [33.1, 63.53],
    #     "SandiaCreekRd": [33.41, 62.76],
    #     "SanmarcosLandfill": [33.09, 62.81],
    #     "SanOnofre": [33.35, 62.47],
    #     "SantaYsabel": [33.11, 63.33],
    #     "Santee": [32.84, 62.98],
    #     "SanYsidro": [32.55, 62.93],
    #     "Sutherland": [33.12, 63.21],
    #     "TierradelSol": [32.65, 63.68],
    #     "WitchCreek": [33.07, 63.26],
    #     "ValleyCenter": [33.22, 62.96],
    # }
    #
    # # create list of local dfs with correct format
    # local_dfs = format_local_dfs(local_path, lat_lon_dict)
    #
    # # concatenate all local data into one dataframe
    # local_df = pd.concat(local_dfs)
    #
    # # merge all global data into one dataframe
    # global_df = merge_global_data(master_df)

    # now create final_df, by merging all into one (only need to do this once, to dump df at specific path)
    # final_df = create_final_df(local_df, global_df)
    #
    # precip_col = final_df.pop("Precip_3hourly")
    # final_df["precipitation"] = precip_col
    # final_df.sort_values(by=["City", "Datetime"], inplace=True)
    # final_df.reset_index(inplace=True, drop=True)
    #
    # # write final_df to pickle file for easier reading
    path = "/Volumes/Transcend/DownscalingData/dataModels/final_df"
    # pickle_dump(final_df, path)

    # load pickle file with final_df
    final_df = pickle_load(path)

    print(final_df.shape)
    print(final_df.columns)
    print(final_df[["Latitude", "Longitude"]])
    # print(final_df["lat"].value_counts())
    # print(final_df["lon"].value_counts())
    # print("lat: ", final_df["Latitude"].describe())
    # print("lon: ", final_df["Longitude"].describe())


if __name__ == "__main__":
    main()
