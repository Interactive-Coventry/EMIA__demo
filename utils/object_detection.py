import warnings
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")

from os.path import join as pathjoin
from os.path import sep
import numpy as np
import pandas as pd
from os import listdir
from natsort import natsorted
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import cv2
import csv
from numpy import random
from pathlib import Path
from libs.foxutils.utils import core_utils
import time
import torch.backends.cudnn as cudnn

from libs.yolov7.models.experimental import attempt_load
from libs.yolov7.utils.datasets import LoadStreams, LoadImages, letterbox
from libs.yolov7.utils.general import check_img_size, check_imshow, \
    non_max_suppression, scale_coords, increment_path
from libs.yolov7.utils.plots import plot_one_box
from libs.yolov7.utils.torch_utils import select_device, load_classifier, TracedModel

import logging
logger = logging.getLogger("utils.object_detection")

vehicle_classes = ["bicycle", "bus", "car", "motorcycle", "person", "truck"]
street_object_classes = ["traffic light", "stop sign", "clock"]

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


def LoadModel(options, device, half, classify=False):
    logger.info(f"Load from {options.weights}")
    model = attempt_load(options.weights, map_location=device)
    stride = int(model.stride.max())
    _ = check_img_size(options.img_size, s=stride)
    if options.trace:
        model = TracedModel(model, device, options.img_size)
    if half:
        model.half()

    if classify:
        modelc = load_classifier(name="resnet101", n=2)  # initialize
        modelc.load_state_dict(torch.load("weights/resnet101.pt", map_location=device)["model"]).to(device).eval()
    else:
        modelc = None

    return model, stride, modelc


class InParams:
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                setattr(self, key, value)


def load_object_detection_model(save_img=True, save_txt=True, device="cuda"):
    classify = False
    if device == "cuda" or device == "gpu":
        device = select_device("0")
        half = device.type != "cpu"
    elif device == "cpu":
        half = False
    else:
        half = True

    opt = InParams(dict(agnostic_nms=False,
                        augment=False,
                        classes=None,
                        conf_thres=0.25,
                        device=device,
                        exist_ok=True,
                        img_size=640,
                        iou_thres=0.45,
                        name="exp",
                        nosave=False,
                        project="runs/detect",
                        save_conf=False,
                        save_txt=save_txt,
                        source="",
                        trace=False,
                        update=False,
                        view_img=False,
                        save_img=save_img,
                        save_dir="",
                        classify=classify,
                        half=half,
                        stride=0,
                        names=[],
                        colors=[],
                        weights=YOLO_MODEL + ".pt"))

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    opt.save_dir = save_dir

    model, stride, modelc = LoadModel(opt, device, half, classify)
    opt.stride = stride
    logger.info(f"New object detection model loaded from Yolov7 on device {device}. Model type {type(model)}.\n")

    opt.names = model.module.names if hasattr(model, "module") else model.names
    opt.colors = [[random.randint(0, 255) for _ in range(3)] for _ in opt.names]
    if opt.half:
        model(torch.zeros(1, 3, opt.img_size, opt.img_size).to(opt.device).type_as(next(model.parameters())))

    return model, opt


def detect_from_image(imgs, od_model, od_opt, device):
    names = od_opt.names
    colors = od_opt.colors
    od_dict_list = []
    od_img = []
    for img in imgs:
        # Padded resize
        od_img = letterbox(img, od_opt.img_size, od_opt.stride)[0]
        # Convert
        od_img = od_img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        od_img = np.ascontiguousarray(od_img)
        od_img = torch.from_numpy(od_img).to(device)
        od_img = od_img.half() if od_opt.half else od_img.float()
        od_img /= 255.0
        if od_img.ndimension() == 3:
            od_img = od_img.unsqueeze(0)
        od_pred = od_model(od_img, augment=od_opt.augment)[0]
        od_pred = non_max_suppression(od_pred, od_opt.conf_thres, od_opt.iou_thres, classes=od_opt.classes,
                                      agnostic=od_opt.agnostic_nms)

        od_dict = {}
        im0 = []
        for i, det in enumerate(od_pred):
            im0 = img.copy()
            if len(det):
                det[:, :4] = scale_coords(od_img.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    od_dict[names[int(c)]] = int(n)

                for *xyxy, conf, cls in reversed(det):
                    label = f"{names[int(cls)]} {conf:.2f}"
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

        od_dict_list.append(od_dict)
        od_img = Image.fromarray(im0[:, :, ::-1])

    return od_img, od_dict_list


def detect(model, opt, image_source=None, file_list=None):
    if file_list is None:
        logger.info(f"Reading images from {image_source}")

    opt.source = image_source
    names = opt.names
    colors = opt.colors
    webcam = opt.source.isnumeric() or opt.source.endswith(".txt") or opt.source.lower().startswith(
        ("rtsp://", "rtmp://", "http://", "https://"))

    vid_path, vid_writer = None, None

    logger.info(f"Is webcam: {webcam}")
    if webcam:
        opt.view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(opt.source, img_size=opt.img_size, stride=opt.stride)
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size, stride=opt.stride, file_list=file_list)

    logger.info(f"Number of images to process: {len(dataset)}")

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        if img is None:
            print(f"None img for path {path}")

        if img is not None:
            img = torch.from_numpy(img).to(opt.device)
            img = img.half() if opt.half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            pred = model(img, augment=opt.augment)[0]
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)

            for i, det in enumerate(pred):
                if webcam:
                    p, s, im0, frame = path[i], "%g: " % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)

                p = Path(p)
                save_path = str(opt.save_dir / p.name)
                txt_path = str(opt.save_dir / "labels" / p.stem) + (
                    "" if dataset.mode == "image" else f"_{frame}")  # img.txt
                s += "%gx%g " % img.shape[2:]
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    class_dict = {}
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                        class_dict[names[int(c)]] = int(n)

                    for *xyxy, conf, cls in reversed(det):
                        if opt.save_txt:
                            with open(txt_path + ".csv", "w", newline="", encoding="utf-8") as csvfile:
                                writer = csv.writer(csvfile)
                                for new_k, new_v in class_dict.items():
                                    writer.writerow([new_k, new_v])

                        if opt.save_img or opt.view_img:
                            label = f"{names[int(cls)]} {conf:.2f}"
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

                if opt.view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)

                if opt.save_img:
                    if dataset.mode == "image":
                        cv2.imwrite(save_path, im0)
                    else:
                        if vid_path != save_path:
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()
                            if vid_cap:
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += ".mp4"
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                        vid_writer.write(im0)

    if opt.save_txt or opt.save_img:
        s = f"\n{len(list(opt.save_dir.glob('labels/*.csv')))} labels saved to {opt.save_dir / 'labels'}" if opt.save_txt else ""
        print(s)

    logger.info(f"Done. ({time.time() - t0:.3f}s)")

    return Image.fromarray(im0[:, :, ::-1])


def detect_from_video(model, opt, stream_link):
    opt.save_img = False
    with torch.no_grad():  # to avoid OOM
        _ = detect(model, opt, image_source=stream_link)
    return


def batch_detect_from_camera_folder(camera_id="1704"):
    logger.info(f"Running on {core_utils.device}.")
    path = "ltaodataservice/Traffic-Imagesv2"
    dataset_dir = pathjoin(core_utils.datasets_dir, "datamall", path.replace("/", sep).replace("?", ""), camera_id, "")
    logger.info(f"Reading images from {dataset_dir}")
    with torch.no_grad():  # to avoid OOM
        _ = detect(image_source=dataset_dir)


def read_classes_from_csv_file(filedir, target_file):
    class_dict = {}
    with open(pathjoin(filedir, target_file), "r", newline="", encoding="utf-8") as csvfile:
        for line in csv.reader(csvfile):
            class_dict[line[0]] = int(line[1])
    return class_dict


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