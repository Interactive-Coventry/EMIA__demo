from os.path import join as pathjoin
from libs.foxutils.utils import core_utils

DEFAULT_IMAGE_FILE = "1703_20230913183132.jpg"
DEFAULT_CAMERA_ID = "1703"

FIXED_CAMERA_SPEC_TABLE_NAME = "fixed_camera_specs"
CAMERA_INFO_PATH = pathjoin("assets", "maps", "camera_ids.csv")

camera_id_key_name = "camera_id"
latitude_key_name = "lat"
longitude_key_name = "lng"

DATASETS_DIR = core_utils.settings["DIRECTORY"]["datasets_dir"]
DEFAULT_DATASET_DIR = pathjoin(DATASETS_DIR, "test", DEFAULT_CAMERA_ID, "")
DEFAULT_FILEPATH = pathjoin(DEFAULT_DATASET_DIR, DEFAULT_IMAGE_FILE)

RUNS_DIR = core_utils.settings["DIRECTORY"]["runs_dir"]
OBJECT_DETECTION_DIR = pathjoin(RUNS_DIR, "detect", "exp", "")
DEFAULT_VEHICLE_FORECAST_FEATURES_DF = pathjoin(DEFAULT_DATASET_DIR, "vf_feature_df.csv")

DEMO_INSTRUCTIONS = ("Click the start button to begin streaming from the selected camera. Click the stop button to end "
                     "the stream. Wait for a few seconds for the dashcam to disconnect, then press refresh. The "
                     "start button will then be active.")

dashcam_ids = ["6206af3f2ac0770155d598c1",
               "5b5649993e120205554b961c",
               ]
dashcam_imeis = ["357730090001398", "351609080169660"]

DASHCAM_IDS = {"Camera " + str(i+1): x for (i, x) in enumerate(dashcam_ids)}
DASHCAM_NAMES = {x: "test_dashcam_" + str(i+1) for (i, x) in enumerate(dashcam_ids)}
DASHCAM_IMEIS = {x: y for (x, y) in zip(dashcam_ids, dashcam_imeis)}

TRAFFIC_IMAGES_PATH = "ltaodataservice/Traffic-Imagesv2"