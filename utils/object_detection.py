from os.path import join as pathjoin
import numpy as np
import pandas as pd
from os import listdir
from natsort import natsorted
from PIL import ImageFile

from libs.foxutils.utils import core_utils
from emia_utils.process_utils import read_classes_from_csv_file

import logging
logger = logging.getLogger("utils.object_detection")
ImageFile.LOAD_TRUNCATED_IMAGES = True

vehicle_classes = ["bicycle", "bus", "car", "motorcycle", "person", "truck"]

WEIGHTS_DIR = pathjoin(core_utils.models_dir, "yolov7", "weights")
YOLO_MODEL = pathjoin(WEIGHTS_DIR, "yolov7_training")
RUNS_FOLDER = pathjoin("runs", "detect", "exp")


def rearrange_class_dict(class_dict, target_classes=None):
    if target_classes is None:
        target_classes = vehicle_classes
    new_dict = {}
    [new_dict.update({x: class_dict[x]}) if x in class_dict.keys() else new_dict.update({x: 0}) for x in target_classes]

    vehicle_list = [new_dict[x] for x in new_dict.keys() if x in vehicle_classes]
    total_pedestrians = new_dict["person"] if "person" in new_dict.keys() else 0
    total_vehicles = sum(vehicle_list) - total_pedestrians
    new_dict["total_pedestrians"] = total_pedestrians
    new_dict["total_vehicles"] = total_vehicles
    return new_dict


def post_process_detect_vehicles(class_dict_list=None, keep_all_detected_classes=True):
    if class_dict_list is None:
        exp_folder = natsorted(listdir(RUNS_FOLDER))[-1]
        filedir = pathjoin(RUNS_FOLDER, exp_folder)
        target_files = [x for x in listdir(filedir) if ".csv" in x]

        if len(target_files) == 0:
            raise ValueError(f"No .csv label files found in directory {filedir}.")
        rows = [read_classes_from_csv_file(filedir, file) for file in target_files]

    else:
        rows = [x for x in class_dict_list]

    detected_classes = np.unique(np.array(core_utils.flatten([list(row.keys()) for row in rows])))

    if keep_all_detected_classes:
        new_rows = [rearrange_class_dict(row, detected_classes) for row in rows]
    else:
        new_rows = [rearrange_class_dict(row) for row in rows]

    class_df = pd.DataFrame.from_dict(new_rows)
    return class_df


def set_results(vehicle_img, object_df):
    return {"image": vehicle_img, "dataframe": object_df}