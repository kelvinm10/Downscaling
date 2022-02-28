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
        curr_df = curr_df[curr_df["bnds"] == 0]
        curr_df.reset_index(inplace=True)
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
def find_closest_coord(lat, lon, target_lat=[30, 32], target_lon=[60, 62.5]):

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
        i.drop(columns=["bnds", "time", "lat", "lon", "time_bnds"], inplace=True)
        list_of_dfs.append(i)

    df_final = reduce(
        lambda left, right: pd.merge(
            left, right, on=["Datetime", "lat_bnds", "lon_bnds"]
        ),
        list_of_dfs,
    )

    return df_final


def main():
    # loop through all data files and load them as a pandas dataframe
    # directory = "/Volumes/Transcend/DownscalingData/ClimateModelData/"
    dataList_path = "/Volumes/Transcend/DownscalingData/dataModels/dataList"
    #
    # dataHolder = load_all_df(directory)
    # pickle_dump(dataHolder, dataList_path)

    # load saved pickle file with all data
    dataList = pickle_load(dataList_path)

    # concatenate all of the corresponding preserving the time series data
    master_df = concat_variables_and_sort(dataList)

    # Create a date column for all global datasets to merge local data with
    for i in master_df:
        create_date_col(i)

    # start to merge global data with local data by date. keep all local data, merge based on local data date
    # need to figure out how to deal with lat / lon lookup and find the closest global model.
    # is it possible to create just 1 `dataset? append all local data, with their closest lat/lon global components

    local_path = "/Volumes/Transcend/DownscalingData/ObservedData3hr/"
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

    local_dfs = []
    for i in os.listdir(local_path):
        # ignore tmp files
        if not i.startswith("._"):
            curr_df = pd.read_csv(local_path + i)
            curr_city = i.split("_")[1].split(".")[0]
            curr_df.insert(0, "City", curr_city)
            # find closest coordinates to one of the 4 values in the global data (lat, lon):
            # 1) (30, 60)
            # 2) (30, 62.5)
            # 3) (32, 60)
            # 4) (32, 62.5)

            closest_coord = find_closest_coord(
                lat_lon_dict[curr_city][0], lat_lon_dict[curr_city][1]
            )
            # insert original lat lon pairs AND closest to global data pairs
            curr_df.insert(2, "Latitude", lat_lon_dict[curr_city][0])
            curr_df.insert(3, "Longitude", lat_lon_dict[curr_city][1])
            curr_df.insert(4, "lat_bnds", closest_coord[0])
            curr_df.insert(5, "lon_bnds", closest_coord[1])
            curr_df["Datetime"] = pd.to_datetime(curr_df["Datetime"])
            local_dfs.append(curr_df)

    # concatenate all local data into one dataframe
    local_df = pd.concat(local_dfs)
    print(local_df.shape)
    # merge all global data into one dataframe
    global_df = merge_global_data(master_df)
    print(global_df.shape)
    # now create final_df, by merging all into one
    final_df = pd.merge(
        local_df, global_df, how="inner", on=["Datetime", "lat_bnds", "lon_bnds"]
    )
    precip_col = final_df.pop("Precip_3hourly")
    final_df["precipitation"] = precip_col
    final_df.sort_values(by=["City", "Datetime"], inplace=True)
    final_df.reset_index(inplace=True, drop=True)
    print(final_df.shape)
    print(final_df.head().to_string())


if __name__ == "__main__":
    main()
