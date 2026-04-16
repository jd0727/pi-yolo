import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from models import YOLO

# from ultralytics import YOLO

if __name__ == '__main__':
    # Create a new YOLO model from scratch
    data_pth = os.path.join(PROJECT_DIR, 'CUSTOM/datasets/exp.yaml')
    save_dir = os.path.join(PROJECT_DIR, 'runs/exp_yv11n_lite')
    # model = YoloV11Main.Large()
    # mfull = YOLO(Cus8Main.Large(num_cls=80, num_dstr=32, img_size=(640, 640)))
    mfull = YOLO(os.path.join(PROJECT_DIR, 'CUSTOM/models/yolo11l.yaml'))
    mfull.train(
        data=data_pth,
        epochs=10,
        imgsz=1024,
        batch=4,
        device='0',
        workers=4,
        amp=False,
        mosaic=0.0,
        save_dir=save_dir
    )

