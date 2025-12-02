import sys
import warnings
import torch
import os

warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

sys.path.insert(0, '/data/capstone-spring-2024/capstone2024/yolov11/ultralytics')

from ultralytics import YOLO


# Load the custom YOLOv8 model
model = YOLO("/data/capstone-spring-2024/capstone2024/yolov11/runs/detect/v11.0.1.n.e300/weights/best.ptd")  # Ensure your .pt file contains the modifications

# Check if the model has been loaded correctly
model.info()

# Export to ONNX using YOLO's built-in export
success = model.export(format="onnx", imgsz=640, simplify=True)

if success:
    print("Model exported successfully to ONNX format")
    
    # Get the ONNX model path
    onnx_model_path = str(model.export()).replace('.pt', '.onnx')
    output_dir = os.path.dirname(onnx_model_path)
    

else:
    print("Failed to export model")