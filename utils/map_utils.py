import pandas as pd
from os.path import join as pathjoin

from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame


def print_expressway_camera_locations(camera_list):
    df_lan = pd.read_csv(pathjoin('assets', 'maps', 'camera_ids.csv'))

    id_lan = [i for (x, i) in zip(df_lan['CameraID'], df_lan.index) if str(x) in camera_list]
    df_lan = df_lan.loc[id_lan]

    geometry = [Point(xy) for xy in zip(df_lan['Longitude'], df_lan['Latitude'])]
    gdf = GeoDataFrame(df_lan, geometry=geometry)

    # this is a simple map that goes with geopandas
    singapore = gpd.read_file(pathjoin('assets', 'maps', 'SGP_adm0.shp'))

    # world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    ax1 = gdf.plot(ax=singapore.plot(figsize=(6, 6)), marker='o', color='red', markersize=15,
                   label='Camera\nlocations')
    # for x, y, label in zip(df_lan['Longitude'], df_lan['Latitude'], df_lan['CameraID']):
    #    ax1.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points")

    sensor = pd.DataFrame({'Longitude': [103.8501], 'Latitude': [1.2897], 'CameraID': ['Weather\nstation']})
    geometry = [Point(xy) for xy in zip(sensor['Longitude'], sensor['Latitude'])]
    gdf = GeoDataFrame(sensor, geometry=geometry)
    gdf.plot(ax=ax1, marker='o', color='yellow', markersize=15)
    for x, y, label in zip(sensor['Longitude'], sensor['Latitude'], sensor['CameraID']):
        ax1.annotate(label, xy=(x, y), xytext=(-25, 5), textcoords="offset points", color='yellow')
    ax1.legend(loc='lower right')
    ax1.axis('off')

    return ax1.get_figure()
