import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from models.yolo.detect.train import DetectionTrainer
from utils import DEFAULT_CFG_DICT
from models.rtdetr.train import RTDETRTrainer

overrides = {
    'task': 'detect', 'mode': 'train',
    'model': os.path.join(PROJECT_DIR, 'CUSTOM/models/yolo12x.yaml'),

    'data': os.path.join(PROJECT_DIR, 'CUSTOM/models/coco.yaml'),
    'save_dir': os.path.join(PROJECT_DIR, 'runs_Lite/coco_yv12x'),

    'name': '',
    ################################################
    'epochs': 50,
    'time': None,
    'patience': 100,
    'device': '0,1,2,3',
    'batch': 32, 'imgsz': 640,
    'save': True, 'save_period': -1,
    'cache': False, 'workers': 0, 'project': None,
    'exist_ok': False,
    'pretrained': None,
    'optimizer': 'auto',
    'verbose': True, 'seed': 0, 'deterministic': True,
    'single_cls': False, 'rect': False, 'cos_lr': False, 'close_mosaic': 5,
    'resume': False, 'amp': False, 'fraction': 1.0, 'profile': False,
    'freeze': None, 'multi_scale': False, 'overlap_mask': True, 'mask_ratio': 4,
    'dropout': 0.0, 'val': True, 'split': 'val', 'save_json': False, 'save_hybrid': False,
    'conf': None, 'iou': 0.7, 'max_det': 300, 'half': False, 'dnn': False, 'plots': True,
    'source': None, 'vid_stride': 1, 'stream_buffer': False, 'visualize': False, 'augment': False,
    'agnostic_nms': False, 'classes': None, 'retina_masks': False, 'embed': None, 'show': False,
    'save_frames': False, 'save_txt': False, 'save_conf': False, 'save_crop': False,
    'show_labels': True, 'show_conf': True, 'show_boxes': True, 'line_width': None,
    'format': 'torchscript', 'keras': False, 'optimize': False, 'int8': False, 'dynamic': False,
    'simplify': True, 'opset': None, 'workspace': None, 'nms': False, 'lr0': 0.01, 'lrf': 0.01,
    'momentum': 0.937, 'weight_decay': 0.0005, 'warmup_epochs': 3.0, 'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1, 'box': 7.5, 'cls': 0.5, 'dfl': 1.5, 'pose': 12.0, 'kobj': 1.0, 'nbs': 64,
    'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5,
    'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.5, 'bgr': 0.0, 'mosaic': 1.0,
    'mixup': 0.0, 'copy_paste': 0.0, 'copy_paste_mode': 'flip', 'auto_augment': 'randaugment',
    'erasing': 0.4, 'crop_fraction': 1.0, 'models': None, 'tracker': 'botsort.yaml',
}

if __name__ == "__main__":
    cfg = DEFAULT_CFG_DICT.copy()
    cfg.update(save_dir='')  # handle the extra key 'save_dir'
    trainer = DetectionTrainer(cfg=cfg, overrides=overrides)
    # trainer = RTDETRTrainer(models=models, overrides=overrides)

    results = trainer.train()
