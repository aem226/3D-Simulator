import sys
import cv2
import random
import numpy as np


sys.path.insert(0, '/data/capstone-spring-2024/capstone2024/yolov11/ultralytics')

from ultralytics import YOLO

class_colors = {         # Color             | Class Name
    0: (65, 105, 225),   # Royal Blue (RGB)  | Car
    1: (144, 238, 144),  # Light Green (RGB) | Van
    2: (216, 191, 216),  # Light Purple (RGB)| Truck
    3: (255, 0, 0),      # Red (RGB)         | Pedestrian
    4: (255, 165, 0),    # Orange (RGB)      | Person Sitting
    5: (255, 255, 0),    # Yellow (RGB)      | Cyclist
    6: (255, 105, 180),  # Pink (RGB)        | Tram (Trolley)
    7: (139, 69, 19),    # Brown (RGB)       | Misc
    8: (211, 211, 211)   # Gray (RGB)        | Don't Care
}


classes = {0: "Car", 1: "Van", 2: "Truck", 3: "Pedestrian", 4: "Person Sitting", 5: "Cyclist", 6: "Tram", 7: "Misc", 8: "Don't Care"}


def draw_bounding_boxes(image, boxes):
   # Generate a unique color for each class ID

   for box in boxes:
      # Extract bounding box coordinates (x1, y1, x2, y2)
      x1, y1, x2, y2 = map(int, box.xyxy[0])  # box.xyxy is an array
      
      # Extract other attributes
      cls = int(box.cls[0])  # Class ID
      conf = float(box.conf[0])  # Confidence score
      angle = float(box.ang[0])  # Angle
      depth = float(box.dep[0])  # Depth
      class_string = classes[cls] # Class String

      # Draw rectangle
      color = class_colors[cls]
      cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

      # Add text label for depth and angle
      label = f"{depth:.2f} {angle:.2f}"
      # cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
      (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

      # Draw background rectangle for the label
      cv2.rectangle(image, (x1, y1 - text_height - 6), (x1 + text_width, y1), color, -1)  # Filled rectangle

      # Draw the label text in white on top of the background
      cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
      
   return image




# Load a model
model = YOLO('/data/capstone-spring-2024/capstone2024/yolov11/runs/detect/v11.0.1.n.e300/weights/best.pt') 

#image = cv2.imread('/data/capstone-spring-2024/all_data/test/images/000025.png')
#image = cv2.imread('/data/capstone-spring-2024/capstone2024/yolov8/output_images/16m.png')
image = cv2.imread('/data/capstone-spring-2024/test_images/2025_Test2.png')

results = model(image)
path = '/data/capstone-spring-2024/capstone2024/yolov11/output_images/2025_Test2.png'
response_data = []

for result in results:
   boxes = result.boxes.cpu().numpy() 

   #processed = draw_bounding_boxes(image, [boxes[0]])
   processed = draw_bounding_boxes(image, boxes)
   
   cv2.imwrite(path, processed)
   

