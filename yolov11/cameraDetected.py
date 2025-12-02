from ultralytics import YOLO
model = YOLO("yolo11n.pt")
print(model.model.args.get("anchors"))