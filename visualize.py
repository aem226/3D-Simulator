import json
import mujoco
import mujoco_viewer
import numpy as np

XML_PATH = "car_scene.xml"
OUTPUT_PATH = "predictions.json"

# load mujoco model
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# load prediction data
with open(OUTPUT_PATH) as f:
    pred = json.load(f)

GT_DISTANCE = pred["gt_depth"]
GT_ANGLE = pred["gt_angle"]

PRED_DISTANCE = pred["pred_depth"]
PRED_ANGLE = pred["pred_angle"]

# reroute bodies
data.body("car_gt").xpos[2] = GT_DISTANCE
data.joint("car_gt_yaw").qpos = np.deg2rad(GT_ANGLE)

data.body("car_pred").xpos[2] = PRED_DISTANCE
data.joint("car_pred_yaw").qpos = np.deg2rad(PRED_ANGLE)

# print for debugging
print("Ground truth:", GT_DISTANCE, "m @", GT_ANGLE, "°")
print("Prediction:", PRED_DISTANCE, "m @", PRED_ANGLE, "°")

# launch interactive viewer
viewer = mujoco_viewer.MujocoViewer(model, data)

while viewer.is_running():
    mujoco.mj_step(model, data)
    viewer.render()

viewer.close()
