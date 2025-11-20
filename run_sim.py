import json
import math

import mujoco
import mujoco_viewer
import numpy as np


XML_PATH = r"C:\Users\aliya\mujoco\mujoco_vs\car_scene.xml"
JSON_PATH = r"C:\Users\aliya\mujoco\mujoco_vs\predictions.json"

#loading the data 
with open(JSON_PATH, "r") as f:
    row = json.load(f)

# row = [class, gt_norm_dist, gt_norm_angle, pred_norm_dist, pred_norm_angle, pred_angle_raw, gt_angle_raw]
cls, gt_nd, gt_na, pd_nd, pd_na, pred_angle_raw, gt_angle_raw = row

DEPTH_SCALE = 60  # convert normalized 0-1 → meters
CLASS_CONFIDENCE = 0.99
CLASS_LABELS = {
    0: "Car",
    1: "Pedestrian",
    2: "Cyclist",
}

GT_D = gt_nd * DEPTH_SCALE
PD_D = pd_nd * DEPTH_SCALE

GT_A = gt_angle_raw       
PD_A = pred_angle_raw      

gt_angle_deg = math.degrees(GT_A)
pd_angle_deg = math.degrees(PD_A)
class_name = CLASS_LABELS.get(cls, f"Class {cls}")

print("\n=== USING VALUES ===")
print("GT distance:", GT_D)
print("GT angle (deg):", gt_angle_deg)
print("Pred distance:", PD_D)
print("Pred angle (deg):", pd_angle_deg)


model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)


def _has_body(name: str) -> bool:
    try:
        model.body(name)
        return True
    except KeyError:
        return False


def _has_joint(name: str) -> bool:
    try:
        model.joint(name)
        return True
    except KeyError:
        return False


# Position the cars only if the bodies exist in the loaded XML
if _has_body("car_gt"):
    data.body("car_gt").xpos[:] = [0, GT_D, 1]
if _has_body("car_pred"):
    data.body("car_pred").xpos[:] = [0, PD_D, 1]

# apply yaw to revolute joints if present
if _has_joint("car_gt_yaw"):
    idx_gt = model.joint("car_gt_yaw").qposadr[0]
    data.qpos[idx_gt] = GT_A
if _has_joint("car_pred_yaw"):
    idx_pred = model.joint("car_pred_yaw").qposadr[0]
    data.qpos[idx_pred] = PD_A


#  view config
viewer = mujoco_viewer.MujocoViewer(model, data)

viewer.cam.fixedcamid = model.camera("main_cam").id
viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
viewer.cam.distance = 35
viewer.cam.elevation = -12
viewer.cam.azimuth = 90

distance_metric = abs(PD_D - GT_D)
orientation_metric = abs(pd_angle_deg - gt_angle_deg)


def _custom_overlay():
    viewer._overlay.clear()

    def set_overlay(gridpos, title, body):
        viewer._overlay[gridpos] = [title, body]

    stats_text = (
        f"distance metric: {distance_metric:.2f} m\n"
        f"orientation metric: {orientation_metric:.1f} deg\n"
        f"class confidence: {CLASS_CONFIDENCE * 100:.0f}%"
    )
    set_overlay(mujoco.mjtGridPos.mjGRID_TOPLEFT, "stats", stats_text)

    set_overlay(
        mujoco.mjtGridPos.mjGRID_TOPRIGHT,
        "reference axis",
        "X / Y / Z",
    )

    gt_text = (
        f"Distance: {GT_D:.1f} m\n"
        f"Orientation: {gt_angle_deg:.1f} deg\n"
        f"Class: {class_name}"
    )
    set_overlay(
        mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
        "Settings / Ground Truth",
        gt_text,
    )

    pred_text = (
        f"Distance: {PD_D:.1f} m\n"
        f"Orientation: {pd_angle_deg:.1f} deg\n"
        f"Class: {class_name}"
    )
    set_overlay(
        mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
        "Prediction",
        pred_text,
    )


viewer._create_overlay = _custom_overlay

# loop
while viewer.is_alive:
    mujoco.mj_step(model, data)
    viewer.render()

viewer.close()
