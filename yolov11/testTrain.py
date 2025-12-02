import sys
import time
import os
from datetime import datetime

Packages = {
   "Latest": '/data/capstone-spring-2024/capstone2024/yolov11',
   "V8_Base":'/data/capstone-spring-2024/capstone2024/yolov8',
   "Orientation":'/data/capstone-spring-2024/codebase/yolov8'
}

#Use the specified ultralytics package
sys.path.insert(0, Packages["Latest"])
#Reload to use the selected package
from ultralytics import YOLO


# Load/Select a model
models = {
#Base Model
   "YOLO11m": 'yolo11m.pt',
   "YOLO11n": 'yolo11n.pt',
   "YOLO11n-obb": 'yolo11n-obb.pt',

#Includes pretrained weights from our capstone team
   "8.1.Depth" : '/data/capstone-spring-2024/capstone2024/yolov8/runs/detect/v8.1.depth/weights/best.pt',
   "Yolov11n.e300" : '/data/capstone-spring-2024/capstone2024/yolov11/runs/detect/v11.0.1.n.e300/weights/best.pt'
}

training_devices = {
   "single":[0],
   "windows":[0,1], #Use two cuda devices (Nvidia GPU)
   "windows2":[2,3], #Use these if the other two GPUs are already training something
   "windows_all":[0,1,2,3] #use all 4 GPUs ONLY if you want to greatly speed up training for a large model
}

model = YOLO(models["YOLO11n"])  # build a new model from scratch OR load an existing model


# Training
start_time = time.time() #Use to calculate time
use_epochs = 1
use_name = 'v11.Test_1_epoch.v0'

results = model.train(
   data='kitti.yaml',
   imgsz=640,
   epochs=use_epochs,
   batch= 16, # -1 gives 32 for one of the GPUs, so 32 is the biggest batchsize
   name=use_name,
   device = training_devices["single"],
   amp = False)


#results



#================================#
#SAVE TRAINING STATS TO TXT FILE
end_time = time.time()
total_time = end_time - start_time
time_per_epoch = total_time / use_epochs 

def format_time(seconds):
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(days)}d {int(hours)}h {int(minutes)}m {int(seconds)}s"


# Extract relevant statistics
stats = {
    "Model Name": use_name,
    "Epochs Used": use_epochs,
    "Total Training Time": format_time(total_time),
    "Estimated Time per Epoch": format_time(time_per_epoch),
    "Date of Completion": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

log_dir = "/data/capstone-spring-2024/capstone2024/yolov11/trainingLogs/" 
log_file = os.path.join(log_dir, use_name + ".txt") 

os.makedirs(log_dir, exist_ok=True)

with open(log_file, "a") as f:
    f.write("\n--- Training Run ---\n")
    for key, value in stats.items():
        f.write(f"{key}: {value}\n")