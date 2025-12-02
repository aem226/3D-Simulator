import sys
import time
import matplotlib.pyplot as plt
import numpy as np

Packages = {
   "Default":'/data/capstone-spring-2024/capstone2024/yolov8',
   "Orientation":'/data/capstone-spring-2024/codebase/yolov8',
   "Latest":'/data/capstone-spring-2024/capstone2024/yolov11',
}

#Use the orientation ultralytics validation
sys.path.insert(0, Packages["Latest"])
from ultralytics import YOLO


# Load a model
models = {
   "nano_v1":'/data/capstone-spring-2024/capstone2024/yolov11/runs/detect/v11.0.1.n.e300/weights/best.pt',
   "Yolov11n.e300":'/data/capstone-spring-2024/capstone2024/yolov11/runs/detect/v11.0.1.n.e300/weights/best.pt',
   "8.1.Depth" : '/data/capstone-spring-2024/capstone2024/yolov8/runs/detect/v8.1.depth/weights/best.pt',
   "8.1.Orientation": '/data/capstone-spring-2024/capstone2024/yolov8/runs/detect/v8.1.orientation3/weights/best.pt'
}

model = YOLO(models["Yolov11n.e300"]) 
use_name = 'v11.0.1.n.e300_troubleshoot1.'


results = model.val(
   data="kitti.yaml",
   imgsz=640,
   save=True,
   name=use_name,
)
