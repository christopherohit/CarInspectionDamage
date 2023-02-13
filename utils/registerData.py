from msilib.schema import Error
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import matplotlib.pyplot as plt
import skimage.io as io

from detectron2.data.datasets import register_coco_instances

current_working = os.getcwd()

try:
    register_coco_instances("car_coco_dataset_train", {}, f"{current_working}/dataset/train/0COCO_train_annos.json", f"{current_working}/dataset/train/")
    register_coco_instances("car_coco_dataset_val", {}, f"{current_working}/dataset/val/0COCO_val_annos.json", f"{current_working}/dataset/val/")
except:
    raise Exception("Format don't right, Please check again")


dataset_dicts = DatasetCatalog.get("car_coco_dataset_train")


def visualize_Data():
    for d in random.sample(dataset_dicts, 3):
        img = io.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("car_coco_dataset_train"), scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        plt.figure(figsize=(10,10))
        plt.imshow(out.get_image()[:, :, ::-1])

