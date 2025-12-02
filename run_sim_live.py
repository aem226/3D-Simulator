import sys
from pathlib import Path
import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer
import cv2

# Add YOLO path
sys.path.insert(0, './yolov11')
from ultralytics import YOLO

# ---------------------------
# Load YOLO model
# ---------------------------
YOLO_MODEL_PATH = "./V11BestWeights.pt"
yolo_model = YOLO(YOLO_MODEL_PATH)
yolo_model.to('cpu')

# ---------------------------
# Load MuJoCo model
# ---------------------------
XML_PATH = "./car_scene.xml"
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# ---------------------------
# Camera setup
# ---------------------------
viewer = mujoco.viewer.launch(model, data)
viewer.cam.fixedcamid = model.camera("main_cam").id
viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

# ---------------------------
# Runtime GT control
# ---------------------------
GT_dist = 12.0
GT_yaw = 0.0

# ---------------------------
# Run YOLO on a frame
# ---------------------------
def run_yolo_on_viewer():
    # Capture the current frame from viewer
    img = viewer.read_pixels(width=640, height=480)  # RGB uint8
    frame_path = "/tmp/mj_viewer_frame.png"
    cv2.imwrite(frame_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    results = yolo_model.predict(source=frame_path, conf=0.25, verbose=False)
    
    df = []
    ih, iw = img.shape[:2]

    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        ang = boxes.ang.cpu().numpy() if hasattr(boxes, "ang") else np.zeros(len(cls))
        dep = boxes.dep.cpu().numpy() if hasattr(boxes, "dep") else np.zeros(len(cls))

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            w = x2 - x1
            h = y2 - y1
            xc = (x1 + w/2) / iw
            yc = (y1 + h/2) / ih
            w /= iw
            h /= ih
            df.append([int(cls[i]), xc, yc, w, h, float(ang[i]), float(dep[i]), float(conf[i])])

    if not df:
        return None
    return pd.DataFrame(df, columns=[
        "class_id", "x_center", "y_center", "width", "height", "rotation_y", "depth", "confidence"
    ])

# ---------------------------
# Overlay function
# ---------------------------
def draw_overlay(viewer, GT_dist, GT_yaw, pred_df):
    viewer._overlay.clear()
    viewer._overlay[mujoco.mjtGridPos.mjGRID_TOPLEFT] = [
        "GT",
        f"Dist: {GT_dist:.2f} m\nYaw: {np.degrees(GT_yaw):.1f}°"
    ]

    if pred_df is not None and len(pred_df) > 0:
        r = pred_df.iloc[0]
        viewer._overlay[mujoco.mjtGridPos.mjGRID_TOPRIGHT] = [
            "YOLO Pred",
            f"Class: {int(r.class_id)}\nDepth: {r.depth:.2f} m\nYaw: {np.degrees(r.rotation_y):.1f}°"
        ]

# ---------------------------
# Main viewer loop
# ---------------------------
frame_counter = 0
while viewer.is_alive:
    # Keyboard controls for GT
    key = viewer.read_key()
    if key == ord('W'):
        GT_dist += 0.2
    if key == ord('S'):
        GT_dist -= 0.2
    if key == ord('A'):
        GT_yaw += 0.05
    if key == ord('D'):
        GT_yaw -= 0.05

    # Update GT body
    data.body("car_gt").xpos[:] = [0, GT_dist, 1.039]
    data.body("car_gt").xquat[:] = mujoco.mju_quatZ(GT_yaw)

    # Step simulation
    mujoco.mj_step(model, data)

    # Run YOLO every 10 frames to save CPU
    pred_df = None
    if frame_counter % 10 == 0:
        try:
            pred_df = run_yolo_on_viewer()
        except Exception as e:
            print("⚠️ YOLO frame error:", e)

    # Update overlays
    draw_overlay(viewer, GT_dist, GT_yaw, pred_df)

    # Render
    viewer.render()
    frame_counter += 1

viewer.close()
