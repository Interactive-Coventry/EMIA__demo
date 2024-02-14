from os.path import join as pathjoin

from libs.foxutils.utils import core_utils

DEFAULT_IMAGE_FILE = "1703_20230913183132.jpg"
DEFAULT_CAMERA_ID = "1703"
EXPRESSWAY_CAMERA_IDS = ["1001",
                         "1002",
                         "1003",
                         "1004",
                         "1005",
                         "1006",
                         "1111",
                         "1112",
                         "1113",
                         "1501",
                         "1502",
                         "1503",
                         "1504",
                         "1505",
                         "1701",
                         "1702",
                         "1703",
                         "1704",
                         "1705",
                         "1706",
                         "1707",
                         "1709",
                         "1711",
                         "2701",
                         "2702",
                         "2703",
                         "2704",
                         "2705",
                         "2706",
                         "2707",
                         "2708",
                         "3702",
                         "3704",
                         "3705",
                         "3793",
                         "3795",
                         "3796",
                         "3797",
                         "3798",
                         "4701",
                         "4702",
                         "4703",
                         "4704",
                         "4705",
                         "4706",
                         "4707",
                         "4708",
                         "4709",
                         "4710",
                         "4712",
                         "4713",
                         "4714",
                         "4716",
                         "4798",
                         "4799",
                         "5794",
                         "5795",
                         "5797",
                         "5798",
                         "5799",
                         "6701",
                         "6703",
                         "6704",
                         "6705",
                         "6706",
                         "6708",
                         "6710",
                         "6711",
                         "6712",
                         "6713",
                         "6714",
                         "6715",
                         "6716",
                         "7791",
                         "7793",
                         "7794",
                         "7795",
                         "7796",
                         "7797",
                         "7798",
                         "8701",
                         "8702",
                         "8704",
                         "8706",
                         "9701",
                         "9702",
                         "9703",
                         "9704",
                         "9705",
                         "9706"]

DATASETS_DIR = core_utils.settings["DIRECTORY"]["datasets_dir"]
DEFAULT_DATASET_DIR = pathjoin(DATASETS_DIR, "test", DEFAULT_CAMERA_ID, "")
DEFAULT_FILEPATH = pathjoin(DEFAULT_DATASET_DIR, DEFAULT_IMAGE_FILE)

RUNS_DIR = core_utils.settings["DIRECTORY"]["runs_dir"]
OBJECT_DETECTION_DIR = pathjoin(RUNS_DIR, "detect", "exp", "")
DEFAULT_VEHICLE_FORECAST_FEATURES_DF = pathjoin(DEFAULT_DATASET_DIR, "vf_feature_df.csv")

DEMO_INSTRUCTIONS = ("Click the start button to begin streaming from the selected camera. Click the stop button to end "
                     "the stream. Wait for a few seconds for the dashcam to disconnect, then press stop again. The "
                     "start button will be enabled after the stop button is pressed twice.")

DASHCAM_IDS = {"Camera 1": "6206af3f2ac0770155d598c1",
               "Camera 2": "6206af3f2ac0770155d598c1"}

TRAFFIC_IMAGES_PATH = "ltaodataservice/Traffic-Imagesv2"
