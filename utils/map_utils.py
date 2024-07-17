import pandas as pd
from os.path import join as pathjoin

from matplotlib import pyplot as plt
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame

from utils.configuration import longitude_key_name, latitude_key_name, camera_id_key_name


def split_in_categories(value, range_size, categories=3):
    if categories == 3:
        if value <  range_size:
            category = 'blue'
        elif  range_size <= value < 2 * range_size:
            category = 'yellow'
        else:
            category = 'red'
        return category
    else:
        raise ValueError("Only 3 categories are supported.")

def categorize_and_color(values, categories=3):
    max_value = max(values)
    range_size = max_value / categories
    colored_values = [split_in_categories(value, range_size, categories) for value in values]
    return colored_values


def assign_heatmap_colors(values, colormap_name='viridis'):
    norm = plt.Normalize(0, max(values))
    cmap = plt.get_cmap(colormap_name)
    colored_values = [cmap(norm(value)) for value in values]
    return colored_values


def print_expressway_camera_locations(camera_info, camera_list, colors=None):
    id_lan = [i for (x, i) in zip(camera_info[camera_id_key_name], camera_info.index) if str(x) in camera_list]
    df_info = camera_info.loc[id_lan]

    geometry = [Point(xy) for xy in zip(df_info[longitude_key_name], df_info[latitude_key_name])]
    gdf = GeoDataFrame(df_info, geometry=geometry)

    # this is a simple map that goes with geopandas
    singapore = gpd.read_file(pathjoin("assets", "maps", "SGP_adm0.shp"))

    has_legend = False
    if colors is None:
        colors = ["red" for _ in range(len(gdf))]
        has_legend = True

    # Plot each point with a different color
    ax1 = singapore.plot(figsize=(6, 6))
    for point, color in zip(gdf.geometry, colors):
        gpd.GeoSeries([point]).plot(ax=ax1, color=color, marker='o', markersize=15)



    #ax1 = gdf.plot(ax=singapore.plot(figsize=(6, 6)), marker="o", color="red", markersize=15,
    #               label="Camera\nlocations")

    # for x, y, label in zip(df_lan["Longitude"], df_lan["Latitude"], df_lan["CameraID"]):
    #    ax1.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points")

    sensor = pd.DataFrame({longitude_key_name: [103.8501], latitude_key_name: [1.2897], camera_id_key_name: ["Weather\nstation"]})
    geometry = [Point(xy) for xy in zip(sensor[longitude_key_name], sensor[latitude_key_name])]
    gdf = GeoDataFrame(sensor, geometry=geometry)
    gdf.plot(ax=ax1, marker="o", color="yellow", markersize=15)
    for x, y, label in zip(sensor[longitude_key_name], sensor[latitude_key_name], sensor[camera_id_key_name]):
        ax1.annotate(label, xy=(x, y), xytext=(-25, 5), textcoords="offset points", color="yellow")

    if has_legend:
        # Add a dummy plot for the legend
        dummy_point_expr = plt.Line2D([0], [0], marker="o", color="red", markersize=5, linewidth=0,
                                 label="Expressway\nCamera")
        dummy_point_mobile = plt.Line2D([0], [0], marker="o", color="blue", markersize=5, linewidth=0,
                                 label="Mobile\nDashcam")
        ax1.legend(handles=[dummy_point_expr, dummy_point_mobile], loc="lower right", fontsize=8)

    ax1.axis("off")

    return ax1.get_figure()
