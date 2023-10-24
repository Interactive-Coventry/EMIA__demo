import collections.abc
import glob
from os.path import join as pathjoin, sep as sep
from random import random

import cv2
import torch
import yaml
from PIL import Image
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from libs.foxutils.utils import core_utils
import sys
import libs.anomalib as anomalib

sys.modules['anomalib'] = anomalib

from libs.anomalib.config import get_configurable_parameters
from libs.anomalib.data.inference import InferenceDataset
from libs.anomalib.models import get_model
from libs.anomalib.post_processing.post_process import (
    superimpose_anomaly_map,
)
from libs.anomalib.utils.callbacks import get_callbacks

torch.set_float32_matmul_precision('medium')

CONFIG_PATHS = pathjoin('libs', 'anomalib', 'models')
MODEL_CONFIG_PAIRS = {
    'patchcore': pathjoin(CONFIG_PATHS, 'patchcore', 'config.yaml'),
    'padim': pathjoin(CONFIG_PATHS, 'padim', 'config.yaml'),
    'cflow':pathjoin(CONFIG_PATHS, 'cflow', 'config.yaml'),
    'dfkde': pathjoin(CONFIG_PATHS, 'dfkde', 'config.yaml'),
    'dfm': pathjoin(CONFIG_PATHS, 'dfm', 'config.yaml'),
    'ganomaly': pathjoin(CONFIG_PATHS, 'ganomaly', 'config.yaml'),
    'stfpm': pathjoin(CONFIG_PATHS, 'stfpm', 'config.yaml'),
    'fastflow': pathjoin(CONFIG_PATHS, 'fastflow', 'config.yaml'),
    'draem': pathjoin(CONFIG_PATHS, 'draem', 'config.yaml'),
    'reverse_distillation': pathjoin(CONFIG_PATHS, 'reverse_distillation', 'config.yaml'),
}

INFER_FOLDER_NAME = 'infer'
HEATMAP_FOLDER_NAME = 'heatmap'
MODELS_DIR = core_utils.models_dir
anomaly_detection_folder = core_utils.settings['MODELS']['anomaly_detection_folder']
anomaly_detection_checkpoint_file = pathjoin(MODELS_DIR, anomaly_detection_folder,
                                             core_utils.settings['MODELS']['anomaly_detection_checkpoint_file'])
anomaly_config_path = pathjoin(MODELS_DIR, anomaly_detection_folder,
                               core_utils.settings['MODELS']['anomaly_detection_config_file'])
anomaly_detection_model = core_utils.settings['MODELS']['anomaly_detection_model']


def load_anomaly_detection_model(model_path=anomaly_detection_checkpoint_file, config_path=anomaly_config_path,
                                 device='gpu'):
    """Run inference."""
    if device == 'cuda':
        device == 'gpu'

    # args = get_args()
    config = get_configurable_parameters(config_path=config_path)
    config.trainer.accelerator = device

    # config.visualization.show_images = args.show
    config.visualization.mode = "simple"
    infer_results_dir = pathjoin(config.project.path, INFER_FOLDER_NAME)
    config.visualization.save_images = True
    config.visualization.image_save_path = infer_results_dir

    model = get_model(config)
    callbacks = get_callbacks(config)

    trainer = Trainer(callbacks=callbacks, **config.trainer)
    model = model.load_from_checkpoint(model_path, hparams=config)
    model.eval()
    print(f'New anomaly detection model loaded from {anomaly_detection_checkpoint_file}.\n')

    return model, trainer, config


def infer(model, trainer, image_size, filepath):
    """Run inference."""

    dataset = InferenceDataset(
        filepath, image_size=tuple(image_size),  # transform_config=transform_config
    )
    dataloader = DataLoader(dataset)

    results = trainer.predict(model=model, dataloaders=[dataloader])
    return results


def get_heatmap(image_filepath, results):
    """Generate heatmap overlay and segmentations, convert masks to images."""

    anomaly_map = results['anomaly_maps'].squeeze().numpy()
    new_dim = anomaly_map.shape
    img = cv2.imread(image_filepath)
    img_opencv = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)

    if anomaly_map is not None:
        heat_map = superimpose_anomaly_map(anomaly_map, img_opencv, normalize=False)
        return heat_map
    else:
        return None


def set_results_anomaly_detection(image_filepath, anomaly_img, results, project_path, orig_dim=None):
    heat_map = get_heatmap(image_filepath, results)
    heat_map_img = Image.fromarray(heat_map, "RGB")
    if orig_dim:
        heat_map_img = heat_map_img.resize(orig_dim)
        anomaly_img = anomaly_img.resize(orig_dim)

    heatmap_results = pathjoin(project_path, HEATMAP_FOLDER_NAME)
    heat_map_path = pathjoin(heatmap_results, image_filepath.split(sep)[-1])
    core_utils.mkdir_if_not_exist(heat_map_path)
    heat_map_img.save(heat_map_path)

    anomaly_label = results['pred_labels'].item()
    anomaly_score = results['pred_scores'].item()
    label_string = 'Anomaly' if anomaly_label else 'Normal'
    return {'image': anomaly_img, 'label': label_string, 'prob': anomaly_score,
            'anomaly_map': results['anomaly_maps'].squeeze().numpy(),
            'heat_map': heat_map, 'heat_map_image': heat_map_img,
            'pred_mask': results['pred_masks'].squeeze().numpy(), 'pred_boxes': results['pred_boxes'],
            'box_scores': results['box_scores'], 'box_labels': results['box_labels']}


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
    with open(new_yaml, 'w') as f:
        yaml.safe_dump(old, f, default_flow_style=False)


def visualize(paths, n_images, is_random=True):
    n_images = min(len(paths), n_images)
    img_list = []
    for i in range(n_images):
        image_name = paths[i]
        if is_random: image_name = random.choice(paths)
        img = cv2.imread(image_name)[:, :, ::-1]
        img_list.append(img)

    return img_list


def show_validation_results(result_path, n_images=5, img_ext='.png'):
    full_path = glob.glob(pathjoin(result_path, 'anomaly', '*' + img_ext), recursive=True)
    img_list_anomaly = visualize(full_path, n_images, is_random=False)
    full_path = glob.glob(pathjoin(result_path, 'normal', '*' + img_ext), recursive=True)
    img_list_normal = visualize(full_path, n_images, is_random=False)

    return img_list_normal, img_list_anomaly
