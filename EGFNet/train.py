import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO



if __name__ == '__main__':
    model = YOLO('/root/lanyun-tmp/ultralytics-yolo11-main/ultralytics/cfg/models/11/EGFNet.yaml')
    # model.load('yolo11n.pt') # loading pretrain weights
    model.train(data='/root/lanyun-tmp/fuben/datasets/GC10-DET-xiugai/data.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=8,
                close_mosaic=0,
                workers=16, 
                optimizer='SGD', # using SGD
                # patience=0, # set 0 to close earlystop.
                #resume=True, # 断点续训,YOLO初始化时选择last.pt
                amp=False, # close amp | loss出现nan可以关闭amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )