import random
import pandas as pd
from os.path import join as pathjoin

from matplotlib import pyplot as plt
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame

CAMERA_INFO_PATH = pathjoin("assets", "maps", "camera_ids.csv")

def get_expressway_camera_info():
    df_lan = pd.read_csv(CAMERA_INFO_PATH)
    df_lan["CameraID"] = [str(x) for x in df_lan["CameraID"]]
    return df_lan


def get_target_camera_info(camera_id):
    df_lan = get_expressway_camera_info()
    df_record = df_lan[df_lan["CameraID"] == str(camera_id)]
    if len(df_record) == 0:
        return False

    df_coord = pd.DataFrame({"camera_id": [camera_id], "lat": [df_record.iloc[0]["Latitude"]],
                             "lng": [df_record.iloc[0]["Longitude"]],
                             "datetime": [None]})
    return df_coord


def print_expressway_camera_locations(camera_list, colors=None):
    df_lan = get_expressway_camera_info()

    id_lan = [i for (x, i) in zip(df_lan["CameraID"], df_lan.index) if str(x) in camera_list]
    df_lan = df_lan.loc[id_lan]

    geometry = [Point(xy) for xy in zip(df_lan["Longitude"], df_lan["Latitude"])]
    gdf = GeoDataFrame(df_lan, geometry=geometry)

    # this is a simple map that goes with geopandas
    singapore = gpd.read_file(pathjoin("assets", "maps", "SGP_adm0.shp"))

    if colors is None:
        colors = ["red" for _ in range(len(gdf))]

    # Plot each point with a different color
    ax1 = singapore.plot(figsize=(6, 6))
    for point, color in zip(gdf.geometry, colors):
        gpd.GeoSeries([point]).plot(ax=ax1, color=color, marker='o', markersize=15)



    #ax1 = gdf.plot(ax=singapore.plot(figsize=(6, 6)), marker="o", color="red", markersize=15,
    #               label="Camera\nlocations")

    # for x, y, label in zip(df_lan["Longitude"], df_lan["Latitude"], df_lan["CameraID"]):
    #    ax1.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points")

    sensor = pd.DataFrame({"Longitude": [103.8501], "Latitude": [1.2897], "CameraID": ["Weather\nstation"]})
    geometry = [Point(xy) for xy in zip(sensor["Longitude"], sensor["Latitude"])]
    gdf = GeoDataFrame(sensor, geometry=geometry)
    gdf.plot(ax=ax1, marker="o", color="yellow", markersize=15)
    for x, y, label in zip(sensor["Longitude"], sensor["Latitude"], sensor["CameraID"]):
        ax1.annotate(label, xy=(x, y), xytext=(-25, 5), textcoords="offset points", color="yellow")

    # Add a dummy plot for the legend
    dummy_point_expr = plt.Line2D([0], [0], marker="o", color="red", markersize=5, linewidth=0,
                             label="Expressway\nCamera")
    dummy_point_mobile = plt.Line2D([0], [0], marker="o", color="blue", markersize=5, linewidth=0,
                             label="Mobile\nDashcam")

    ax1.legend(handles=[dummy_point_expr, dummy_point_mobile], loc="lower right", fontsize=8)
    ax1.axis("off")

    return ax1.get_figure()
