import os
import sys

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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


def pretty_xml_node(node, indent='\t', newline='\n', level=0):
    if node:
        # 如果element的text没有内容
        if node.text == None or node.text.isspace():
            node.text = newline + indent * (level + 1)
        else:
            node.text = newline + indent * (level + 1) + node.text.strip() + newline + indent * (level + 1)
    # 此处两行如果把注释去掉，Element的text也会另起一行
    # else:
    # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(node)  # 将elemnt转成list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        pretty_xml_node(subelement, indent, newline, level=level + 1)
    return True


def dict2node(full_dict, node=None):
    node = ET.Element(node) if isinstance(node, str) else node
    if isinstance(full_dict, dict):
        node.text = '\r\n' if node.text is None else node.text + '\r\n'
        node.tail = '\r\n' if node.tail is None else node.tail + '\r\n'
        for key, val in full_dict.items():
            sub_node = ET.SubElement(node, key) if node.find(key) is None else node.find(key)
            dict2node(val, node=sub_node)
    else:
        node.text = str(full_dict)
    return node


def creat_xml(xml_pth, img_size, xyxys, cinds, confs, cls_mapper):
    anno_dict = {
        'folder': 'JPEGImages',
        'filename': os.path.basename(img_pth),
        'source': {'database': 'Unknown'},
        'size': {'width': img_size[0], 'height': img_size[1], 'depth': 3},
        'segmented': 0
    }
    root = dict2node(anno_dict, node='annotation')
    for xyxy, cind, conf in zip(xyxys, cinds, confs):
        obj = ET.SubElement(root, 'object')
        xyxy = np.round(xyxy).astype(np.int32)
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xyxy[0])
        ET.SubElement(bndbox, 'ymin').text = str(xyxy[1])
        ET.SubElement(bndbox, 'xmax').text = str(xyxy[2])
        ET.SubElement(bndbox, 'ymax').text = str(xyxy[3])
        # ET.SubElement(obj, 'name').text = box['name']
        ET.SubElement(obj, 'name').text = cls_mapper[int(cind)]
        ET.SubElement(obj, 'conf').text = str(conf)
    pretty_xml_node(root)
    root = ET.ElementTree(root)
    root.write(xml_pth, encoding='utf-8')
    return root


def load_txt(file_pth: str, extend: str = 'txt', encoding: str = 'utf-8'):
    with open(file_pth, 'r', encoding=encoding) as file:
        lines = file.readlines()
    lines = [line.replace('\n', '') for line in lines]
    return lines


def ensure_folder_pth(file_pth: str) -> str:
    if not os.path.exists(file_pth):
        os.makedirs(file_pth)
    return file_pth


def load_yaml(file_pth: str, encoding: str = 'utf-8'):
    with open(file_pth, 'r', encoding=encoding) as file:
        dct = yaml.safe_load(file)
    return dct


def nms(xyxys, scores, cinds, conf_thres=0.2, iou_thres=0.4, max_det=300):
    xydwhs = np.concatenate([xyxys[:, :2], xyxys[:, 2:4] - xyxys[:, :2]], axis=1)
    indices = cv2.dnn.NMSBoxes(xydwhs, scores, conf_thres, iou_thres)
    xyxys = np.array(xyxys)[indices]
    scores = np.array(scores)[indices]
    cinds = np.array(cinds)[indices]
    if len(xyxys) > max_det:
        order = np.argsort(-scores)[:max_det]
        xyxys = xyxys[order]
        scores = scores[order]
        cinds = cinds[order]
    return xyxys, scores, cinds


def infer_at_img(model, img_pth, xml_pth, conf=0.4, iou_thres=0.6, max_det=300):
    im1 = Image.open(img_pth).convert('RGB')
    img_size = im1.size
    result = model.predict(
        source=im1,
        imgsz=img_size,
        save=False,
        conf=conf
    )
    mapper = result[0].names
    boxes_confs_cinds = result[0].boxes.data.detach().cpu().numpy()
    boxes, confs, cinds = boxes_confs_cinds[:, :4], boxes_confs_cinds[:, 4], boxes_confs_cinds[:, 5]

    boxes_all, confs_all, cinds_all = nms(
        boxes, confs, cinds, conf_thres=conf, iou_thres=iou_thres, max_det=max_det)
    creat_xml(xml_pth, img_size, xyxys=boxes_all, confs=confs_all, cinds=cinds_all, cls_mapper=mapper)
    return True


if __name__ == '__main__':
    wei_pth = os.path.join(PROJECT_DIR, 'runs_DT/voc_yolo11l_tr7/weights/last.pt')
    # wei_pth = os.path.join(PROJECT_DIR, 'runs/distrinet_yv11e/weights/best.pt')

    device = torch.device('cuda:1')
    model = YOLO(wei_pth)
    model.to(device)
    # model = DetectionModel(wei_pth)

    root = '/home/data-storage/VOC'
    # root = '/home/data-storage/JD/TDUnion'

    metas = load_txt(os.path.join(root, 'ImageSets/Main', '0712trainval_auto.txt'))
    # metas = [fname.split('.')[0] for fname in sorted(os.listdir(os.path.join(root, 'JPEGImages')))]
    # metas = ['020297_4596_1536_5472_3078']

    xml_dir = ensure_folder_pth(os.path.join(root, 'Annotations_pd7'))
    img_dir = os.path.join(root, 'JPEGImages')
    metas = [m for m in metas if len(m) > 0]
    for i, meta in enumerate(sorted(metas)):
        img_pth = os.path.join(img_dir, meta + '.jpg')
        xml_pth = os.path.join(xml_dir, meta + '.xml')
        # if os.path.exists(xml_pth):
        #     continue
        print(meta)
        # if not os.path.exists(xml_pth):
        infer_at_img(model, img_pth, xml_pth, conf=0.4, iou_thres=0.6, max_det=300)
