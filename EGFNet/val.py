import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('/root/lanyun-tmp/ultralytics-yolo11-main/runs/prune/yolov11-finetune10/weights/best.pt') # 选择训练好的权重路径
    model.val(data='/root/lanyun-tmp/fuben/datasets/GC10-DET-xiugai/data.yaml',
              split='val', # split可以选择train、val、test 根据自己的数据集情况来选择.
              imgsz=640,
              batch=16,
              # iou=0.7,
              # rect=False,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )