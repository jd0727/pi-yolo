import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from models import YOLO
from models.yolo.detect import DetectionTrainer
from engine.model import Model
from nn import DetectionModel

from PIL import Image

import yaml

import xml.etree.ElementTree as ET
import torch.nn.functional as F
import cv2
import numpy as np
import torch


# from ultralytics import YOLO


def load_yaml(file_pth: str, encoding: str = 'utf-8'):
    with open(file_pth, 'r', encoding=encoding) as file:
        dct = yaml.safe_load(file)
    return dct


if __name__ == '__main__':
    wei_pth = os.path.join(PROJECT_DIR, 'runs/best.pt')
    data_pth = os.path.join(PROJECT_DIR, 'CUSTOM/models/distrinet.yaml')
    # device = torch.device('cuda:3')
    model = YOLO(wei_pth)
    # model.to(device)
    # model = DetectionModel(wei_pth)

    # DetectionValidator
    model.val(
        data=data_pth,
        imgsz=1024,
        batch=8,
        device='1',
        workers=8,
        amp=False,
        save_dir=os.path.join(PROJECT_DIR, 'runs/val')
    )
