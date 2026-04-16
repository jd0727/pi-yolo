# PI-YOLO: An Efficient Detector for Power Line Inspection using Aerial Images

This project is developed based on the YOLO project (https://github.com/ultralytics/ultralytics). The following modifications have been made to this project:
* Use datasets in VOC format for training instead of the original YOLO format
* Add PI-YOLO related codes (./CUSTOM, ./nn/modules/cus_add.py, ...)
* Add the function to save evaluation results for each category during training

To use PI-YOLO, you can:
* Migrate PI-YOLO related codes to the original YOLO project, and conduct training and inference in its original form
* Follow the instructions below to train PI-YOLO on datasets in VOC format

## Dataset Preparation
The dataset must be in the standard VOC format:
```
root
├─ JPEGImages
├─ Annotations
└─ ImageSets
    └─ Main
        ├─ train.txt
        └─ val.txt
```
For the corresponding configuration file `data.yaml`:
* The `path` field points to the `ImageSets/Main` folder of the dataset
* The `train`, `val` and other fields point to the TXT files under `path`, such as `train.txt` and `val.txt`
* The `JPEGImages` and `Annotations` folders are determined by the `path` field and cannot be specified additionally

## Single GPU Training
```
python CUSTOM/train.py
```
## Multi GPU Training
```
CUDA_VISIBLE_DEVICES=0,1  torchrun --nproc_per_node=2 --nnodes=1 --master_port 1638 train_muti_coco.py
```
## Single-scale Inference
Resize the original image to the specified scale, perform inference and generate XML annotations:
```
python CUSTOM/infer.py
```
## Multi-scale Inference
Resize the original image to multiple scales, fuse the results after inference, and generate XML annotations:
```
python CUSTOM/infer_ms.py
```
## Notes
* The assets folder contains some examples of inspection images
* For high-resolution inspection images, you need to first crop them into a dataset of about 1024 pixels, train on the cropped dataset, and then perform multi-scale inference on the original images
* The pretrained folder contains the pre-trained weights of PI-YOLO on PIUD. The network can predict 53 classes, among which 'insulator_comp_break' is deprecated, and the remaining 52 classes correspond to the paper

