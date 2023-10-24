import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# config unique dir
#"Metric `PrecisionRecallCurve` will save all targets and predictions in buffer.*")

import collections.abc
import glob
import sys
from os.path import join as pathjoin
from random import random
import cv2
import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import libs.anomalib as anomalib
sys.modules["anomalib"] = anomalib

from libs.anomalib.config import get_configurable_parameters
from libs.anomalib.data.inference import InferenceDataset
from libs.anomalib.models import get_model
from libs.anomalib.post_processing.post_process import (
    superimpose_anomaly_map,
)
from libs.anomalib.utils.callbacks import get_callbacks, MinMaxNormalizationCallback
from libs.anomalib.data.utils import get_transforms
from libs.foxutils.utils import core_utils

import logging
logger = logging.getLogger("utils.anomaly_detection")

torch.set_float32_matmul_precision("medium")

CONFIG_PATHS = pathjoin("libs", "anomalib", "models")
MODEL_CONFIG_PAIRS = {
    "patchcore": pathjoin(CONFIG_PATHS, "patchcore", "config.yaml"),
    "padim": pathjoin(CONFIG_PATHS, "padim", "config.yaml"),
    "cflow": pathjoin(CONFIG_PATHS, "cflow", "config.yaml"),
    "dfkde": pathjoin(CONFIG_PATHS, "dfkde", "config.yaml"),
    "dfm": pathjoin(CONFIG_PATHS, "dfm", "config.yaml"),
    "ganomaly": pathjoin(CONFIG_PATHS, "ganomaly", "config.yaml"),
    "stfpm": pathjoin(CONFIG_PATHS, "stfpm", "config.yaml"),
    "fastflow": pathjoin(CONFIG_PATHS, "fastflow", "config.yaml"),
    "draem": pathjoin(CONFIG_PATHS, "draem", "config.yaml"),
    "reverse_distillation": pathjoin(CONFIG_PATHS, "reverse_distillation", "config.yaml"),
}

INFER_FOLDER_NAME = "infer"
HEATMAP_FOLDER_NAME = "heatmap"
MODELS_DIR = core_utils.models_dir
anomaly_detection_folder = core_utils.settings["ANOMALY_DETECTION"]["anomaly_detection_folder"]
anomaly_detection_checkpoint_file = pathjoin(MODELS_DIR, anomaly_detection_folder,
                                             core_utils.settings["ANOMALY_DETECTION"]["anomaly_detection_checkpoint_file"])
anomaly_config_path = pathjoin(MODELS_DIR, anomaly_detection_folder,
                               core_utils.settings["ANOMALY_DETECTION"]["anomaly_detection_config_file"])
anomaly_detection_model = core_utils.settings["ANOMALY_DETECTION"]["anomaly_detection_model"]


def load_anomaly_detection_model(model_path=anomaly_detection_checkpoint_file, config_path=anomaly_config_path,
                                 device="gpu", set_up_trainer=True):
    """Run inference."""
    config = get_configurable_parameters(config_path=config_path)
    config.trainer.accelerator = "gpu" if device == "gpu" else "cpu"

    # config.visualization.show_images = args.show
    config.visualization.mode = "simple"
    infer_results_dir = pathjoin(config.project.path, INFER_FOLDER_NAME)
    config.visualization.save_images = True
    config.visualization.image_save_path = infer_results_dir

    model = get_model(config)
    model = model.load_from_checkpoint(model_path, hparams=config)
    model.eval()
    logger.info(f"New anomaly detection model loaded from {anomaly_detection_checkpoint_file} on device {device}. Model type {type(model)}.\n")

    if set_up_trainer:
        callbacks = get_callbacks(config)
        trainer = Trainer(callbacks=callbacks, **config.trainer)
        return model, trainer, config
    else:
        ad_tfms = get_transforms(image_size=tuple(config.dataset.image_size), config=config.dataset.transform_config.eval)
        return model, ad_tfms, config


def infer_from_image(img, model, device, ad_tfms):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocess
    processed_image = ad_tfms(image=img)["image"]
    if len(processed_image) == 3:
        processed_image = processed_image.unsqueeze(0)
    processed_image.to(device)

    # Predict
    batch = {"image": torch.cat([processed_image], 0)}
    outputs = model.predict_step(batch, 0, 0)

    # Post Process
    norm_callback = MinMaxNormalizationCallback()
    norm_callback._normalize_batch(outputs, model)
    return outputs


def infer(model, trainer, image_size, filepath):
    """Run inference."""

    dataset = InferenceDataset(
        filepath, image_size=tuple(image_size),  # transform_config=transform_config
    )
    dataloader = DataLoader(dataset)

    results = trainer.predict(model=model, dataloaders=[dataloader])
    return results


def get_heatmap(image_filepath, results):
    """Generate heatmap overlay and segmentations, convert masks to images.

    Args:
        image_filepath (str | cv2 image): Path to the image file OR cv2 image.
        results (dict): Results from the inference.
    """

    if isinstance(image_filepath, str):
        img = cv2.imread(image_filepath)
    else:
        img = image_filepath

    anomaly_map = results["anomaly_maps"].detach().squeeze().numpy()
    new_dim = anomaly_map.shape

    img_opencv = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)
    if anomaly_map is not None:
        heat_map = superimpose_anomaly_map(anomaly_map, img_opencv, normalize=False)
        return heat_map
    else:
        return None


def set_results(orig_img, results, orig_dim=None):
    """Set results for anomaly detection.

    Args:
        orig_img (str | cv2 image): Path to the image file OR cv2 image.
        results (dict): Results from the inference.
        :param orig_img:
        :param results:
        :param orig_dim:
    """

    anomaly_img = results["image"].detach().squeeze().numpy()
    anomaly_img = np.transpose(anomaly_img, (1, 2, 0))
    anomaly_img = cv2.cvtColor(anomaly_img, cv2.COLOR_BGR2RGB)

    heat_map = get_heatmap(orig_img, results)
    if orig_dim:
        heat_map = cv2.resize(heat_map, orig_dim, interpolation=cv2.INTER_AREA)
        anomaly_img = cv2.resize(anomaly_img, orig_dim, interpolation=cv2.INTER_AREA)

    anomaly_label = results["pred_labels"].item()
    anomaly_score = results["pred_scores"].item()
    label_string = "Anomaly" if anomaly_label else "Normal"
    return {"image": anomaly_img, "label": label_string, "prob": anomaly_score,
            "anomaly_map": results["anomaly_maps"].detach().squeeze().numpy(),
            "heat_map": heat_map,
            "pred_mask": results["pred_masks"].squeeze().numpy(), "pred_boxes": results["pred_boxes"],
            "box_scores": results["box_scores"], "box_labels": results["box_labels"]}


def update_yaml(old_yaml, new_yaml, new_update):
    # load yaml
    with open(old_yaml) as f:
        old = yaml.safe_load(f)

    def update(d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    old = update(old, new_update)

    # save the updated / modified yaml file
    with open(new_yaml, "w") as f:
        yaml.safe_dump(old, f, default_flow_style=False)


def visualize(paths, n_images, is_random=True):
    n_images = min(len(paths), n_images)
    img_list = []
    for i in range(n_images):
        image_name = paths[i]
        if is_random:
            image_name = random.choice(paths)
        img = cv2.imread(image_name)[:, :, ::-1]
        img_list.append(img)

    return img_list


def show_validation_results(result_path, n_images=5, img_ext=".png"):
    full_path = glob.glob(pathjoin(result_path, "anomaly", "*" + img_ext), recursive=True)
    img_list_anomaly = visualize(full_path, n_images, is_random=False)
    full_path = glob.glob(pathjoin(result_path, "normal", "*" + img_ext), recursive=True)
    img_list_normal = visualize(full_path, n_images, is_random=False)

    return img_list_normal, img_list_anomaly
