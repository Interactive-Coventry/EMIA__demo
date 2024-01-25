import json
from os.path import join as pathjoin
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
import torch
import torch.nn as nn
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from torch.optim import lr_scheduler
from torchvision.models import resnet18, resnet50

from libs.foxutils.utils import train_with_lightning, core_utils
from libs.foxutils.utils.lightning_models.prediction_model import PredictionModel

import logging
logger = logging.getLogger("utils.weather_detection_utils")

device = core_utils.device
default_models_dir = pathjoin(core_utils.models_dir, "EMIA", "weather_from_image")
weather_dict = {"Clear": 0, "Clouds": 1, "Rain": 2, "Thunderstorm": 3}
weather_classes = {v: k for k, v in weather_dict.items()}

weather_description_dict = {"broken clouds": 0, "clear sky": 1, "heavy intensity rain": 2,
                            "light intensity shower rain": 3, "light rain": 4, "moderate rain": 5,
                            "scattered clouds": 6, "thunderstorm": 7, "thunderstorm with heavy rain": 8,
                            "thunderstorm with light rain": 9, "thunderstorm with rain": 10}
weather_description_classes = {v: k for k, v in weather_description_dict.items()}

tfms = transforms.Compose([transforms.Resize((224, 224)),
                           transforms.ToTensor(),
                           # For pretraining on ImageNet
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                           ])
inv_tfms = transforms.Compose([transforms.Normalize([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
                                                    [1 / 0.229, 1 / 0.224, 1 / 0.255])])


class ImageDataset(Dataset):
    def __init__(self, df, img_dir, labeling_function, transform=None, target_transform=None, inv_transform=None):
        self.filenames = df["file"]
        self.folders = df["folder"]
        self.img_dir = img_dir
        self.transform = transform
        self.inv_transform = inv_transform
        self.target_transform = target_transform
        self.data_df = df
        self.labeling_function = labeling_function

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = pathjoin(self.img_dir, self.folders.iloc[idx], self.filenames.iloc[idx])
        img = Image.open(img_path)  # read_image(img_path) # reads to torch
        image = T.PILToTensor()(img)
        label = self.labeling_function(self.data_df.iloc[idx])

        if self.transform:
            image = self.transform(img)  # .unsqueeze(0)

        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def read_image_and_get_label(df, img_dir, filename, labeling_function, label_classes):
    img_path = pathjoin(img_dir, filename)
    img = read_image(img_path).squeeze()
    data_row = df[df["file"] == filename].iloc[0]
    print(data_row)
    label = label_classes[labeling_function(data_row)]

    pil_img = T.ToPILImage()(img)
    plt.imshow(np.asarray(pil_img))
    plt.show()

    # display(pil_img)
    print(f"Label: {label}")


def prepare_image_for_model_input(image_dataset_dir, filename, show_preprocessed=True):
    if show_preprocessed:

        fig, (ax1, ax2) = plt.subplots(1, 2)

        img_file = pathjoin(image_dataset_dir, filename)
        img = Image.open(img_file)
        ax1.imshow(img)
        ax1.set_title("Original")

        img = tfms(img).unsqueeze(0)
        pil_img = T.ToPILImage()(img.squeeze())
        ax2.imshow(pil_img)
        ax2.set_title("Preprocessed")
        # print(img.shape) # torch.Size([1, 3, 224, 224])
        plt.show()

    else:
        fig, (ax1) = plt.subplots(1, 1)

        img_file = pathjoin(image_dataset_dir, filename)
        img = Image.open(img_file)
        ax1.imshow(img)
        ax1.set_title("Original")
        img = tfms(img).unsqueeze(0)
        plt.show()

    return img


def get_imagenet_labels_map():
    # Load ImageNet class names
    labels_map_file = pathjoin(core_utils.models_dir, "label_maps", "ImageNet", "labels_map.txt")
    labels_map = json.load(open(labels_map_file))
    labels_map = [labels_map[str(i)] for i in range(1000)]
    return labels_map


def predict(img, model=None, labels_map=None, model_name="efficientnet-b7"):
    if model_name == "efficientnet-b7" and labels_map is None:
        labels_map = get_imagenet_labels_map()
        if model is None:
            model = EfficientNet.from_pretrained("efficientnet-b7")

    img = tfms(img).unsqueeze(0)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(img.to(device))

    idx = torch.topk(outputs, k=1).indices.squeeze(0).tolist()[0]
    labels = labels_map[idx]
    prob = torch.softmax(outputs, dim=1)[0, idx].item() * 100
    return labels, prob, outputs

def predict_weather_class(image_dataset_dir, filename, model=None, labels_map=None, model_name="efficientnet-b7"):
    img_file = pathjoin(image_dataset_dir, filename)
    img = Image.open(img_file)
    labels, prob, outputs = predict(img, model, labels_map, model_name)

    num_show = min(5, len(labels_map))
    logger.debug(f"\nPredictions with {model_name}-----")
    for idx in torch.topk(outputs, k=num_show).indices.squeeze(0).tolist():
        prob = torch.softmax(outputs, dim=1)[0, idx].item()
        logger.debug("{label:<75} ({p:.2f}%)".format(label=labels_map[idx], p=prob * 100))

    return labels, prob


def calculate_acc(model, dataloaders, dataset_sizes,
                  forward_fun=lambda x, y: train_with_lightning.get_argmax_label(x, y)[1], phase="train"):
    running_corrects = 0

    model = model.to(device)
    model.eval()

    # Iterate over data.
    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = forward_fun(model, inputs)

        running_corrects += torch.sum(preds == labels.data)

    epoch_acc = running_corrects.double() / dataset_sizes[phase]
    logger.debug(f"{phase} Acc: {epoch_acc:.4f}")


def get_model_name(version, base_model_name, has_augmentation=False, has_transfer_learning=None):
    model_name = base_model_name + "-v" + str(version)
    if has_transfer_learning:
        model_name = model_name + "_TL"
    if has_augmentation:
        model_name = model_name + "_DA"

    model_filename = model_name
    return model_name, model_filename


def get_params_from_model_name(model_name):
    has_transfer_learning = "_TL" in model_name
    has_augmentation = "_DA" in model_name
    base_model_name = model_name.split("_")[0]
    base_model_name = base_model_name.split(".")[0]
    version = base_model_name.split("-v")[-1]
    base_model_name = model_name.split("-v")[0]
    return base_model_name, version, has_augmentation, has_transfer_learning


def load_weather_detection_model():
    MODELS_DIR = core_utils.models_dir
    weather_detection_folder = core_utils.settings["WEATHER_DETECTION"]["weather_detection_folder"]
    weather_detection_checkpoint_file = core_utils.settings["WEATHER_DETECTION"]["weather_detection_checkpoint_file"]
    weather_detection_model = core_utils.settings["WEATHER_DETECTION"]["weather_detection_model"]

    checkpoint_path = pathjoin(MODELS_DIR, weather_detection_folder, weather_detection_checkpoint_file + ".pts")
    weather_class_model_name, version, has_augmentation, has_transfer_learning = get_params_from_model_name(
        weather_detection_checkpoint_file)

    weather_class_model, weather_class_model_name = make(weather_class_model_name, checkpoint_path, weather_dict,
                                                         version, has_augmentation=has_augmentation,
                                                         has_transfer_learning=has_transfer_learning)
    logger.info(f"New weather classification model loaded from {weather_detection_checkpoint_file}. Model type {type(weather_class_model)}.\n")
    return weather_class_model, weather_class_model_name


def make(base_model_name, weight_path, class_mapping, version, has_augmentation=False, has_transfer_learning=True):
    if base_model_name == "resnet-50":
        base_model = resnet50(weights="IMAGENET1K_V1")  # weights=None, weights="IMAGENET1K_V1"
        has_transfer_learning = True
    elif base_model_name == "resnet-18":
        base_model = resnet18(weights="IMAGENET1K_V1")  # weights=None, weights="IMAGENET1K_V1"
        has_transfer_learning = True
    elif base_model_name == "efficientnet-b7":
        base_model = EfficientNet.from_pretrained("efficientnet-b7")

    if has_transfer_learning:
        for param in base_model.parameters():
            param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = base_model.fc.in_features
    base_model.fc = nn.Linear(num_ftrs, len(class_mapping))

    base_model = base_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_conv = optim.SGD(base_model.fc.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=10, gamma=0.1)
    task = "multiclass"
    model_name, model_filename = get_model_name(version, base_model_name=base_model_name,
                                                has_augmentation=has_augmentation,
                                                has_transfer_learning=has_transfer_learning)
    logger.info(f"\nRunning for a {model_name} model...")

    forward_function = train_with_lightning.get_label_probs  # train_with_lightning.get_argmax_label

    target_model_class = PredictionModel
    model_params = dict(model_class=base_model,
                        task=task,
                        num_classes=len(class_mapping),
                        num_labels=len(class_mapping),
                        forward_function=forward_function,
                        loss_fun=criterion,
                        configure_optimizers_fun=lambda: {"optimizer": optimizer_conv,
                                                          "lr_scheduler": exp_lr_scheduler},
                        average="weighted",
                        has_augmentation=has_augmentation,
                        class_mapping=class_mapping)

    model_ld = train_with_lightning.pl_load_trained_model(target_model_class, weight_path, **model_params)

    return model_ld, model_filename

def set_results(label, prob):
    weather_label = "{label:<75} ({p:.2f}%)".format(label=label, p=prob)
    return {"label": label, "prob": prob / 100, "label_string": weather_label}
