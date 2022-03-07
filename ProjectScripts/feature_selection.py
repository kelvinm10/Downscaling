import os

import pandas as pd
import plotly.express as px
from preprocessing import pickle_load
import numpy as np
from scipy import stats
import itertools
import statsmodels.api
import plotly.figure_factory as ff



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
            holder.append(np.round(stats.pearsonr(dataframe[i].values, dataframe[j].values)[0], 3))
        result.append(holder)

    print(result)
    fig = ff.create_annotated_heatmap(
        result, x=continuous, y=continuous, showscale=True, colorscale="Blues"
    )
    fig.update_layout(title="Continuous-Continuous Correlation Matrix")
    fig.show()


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

def plot_continuous(df, series, response):
    # if response is continuous, scatterplot with trend line
    if is_continuous(df, response):
        fig = px.scatter(df, x=series.name, y=response, trendline="ols")

    # if response is categorical, violin plot
    else:
        fig = px.violin(df, x=response, y=series.name, box=True)
    return fig

def cont_cont_correlation(dataframe, list_of_predictors, response):
    os.mkdir("finalTables")
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
    predictors = df.columns[6:21]
    conditions = [
        df["precipitation"] == 0,
        df["precipitation"] > 0
    ]
    outputs = [0,1]
    res = np.select(conditions, outputs)
    df["precip_binary"] = pd.Series(res)

    df = df.dropna()
    agua = "AguaCaliente"
    agua_df = df[df["City"] == agua]
    #result = cont_cont_correlation(agua_df, agua_df.columns[4:],"precipitation")
    holder = list(agua_df.columns[7:21])
    holder.append("precipitation")

    cont_cont_heatmap(list(df.columns[7:]), df)












if __name__ == "__main__":
    main()