import argparse
import cv2
import depthai as dai
import numpy as np
import time
import math
import torch
#import json
import json
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Union, Optional
from ultralytics.utils import LOGGER

def get_model_config(model_name): 
    # Returns paths and metadata (e.g., input shape, blob path) for different models, including custom YOLOs.
    model_configs = {
        'custom_yolo': {
            'blobPath': '/data/capstone-spring-2024/capstone2024/yolov11/runs/detect/v11.0.1.n.e300/weights/best.blob',
            'nnShape': (640, 640),
            'maxOutputFrameSize': 2457600,
            'type': 'yolo',
            'configPath': '/data/capstone-spring-2024/capstone2024/yolov11/pipeline_config.json'
        } 
    }
    return model_configs[model_name]

def xywh2xyxy(x):
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y

def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
    rotated=False,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes.
    For a 6-column blob output (xywh, conf, class) the function directly returns the detections.
    """
    import torchvision  # scope for faster 'import ultralytics'

    # If the blob outputs 6 columns (typical for many detectors), use the fast branch.
    if prediction.shape[-1] == 6:  # expected format: [x, y, w, h, conf, class]
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    # (Legacy code for more complex outputs has been removed)
    raise ValueError("Unexpected prediction shape. Expected 6 columns for [xywh, conf, class].")

@dataclass
class Detection:
    # Updated Detection class without angle and depth
    img_detection: Union[None, dai.ImgDetection, dai.SpatialImgDetection]
    label: int
    confidence: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float

class GenericNNOutput:
    """
    Generic NN output, to be used for higher-level abstractions.
    """
    def __init__(self, nn_data: Union[dai.NNData, dai.ImgDetections, dai.SpatialImgDetections]):
        self.nn_data = nn_data

    def getTimestamp(self) -> timedelta:
        return self.nn_data.getTimestamp()

    def getSequenceNum(self) -> int:
        return self.nn_data.getSequenceNum()

@dataclass
class Detections(GenericNNOutput):
    """
    Detection results containing bounding boxes, labels and confidences.
    """
    def __init__(self,
                 nn_data: Union[dai.NNData, dai.ImgDetections, dai.SpatialImgDetections]):
        GenericNNOutput.__init__(self, nn_data)
        self.detections: List[Detection] = []

def decode(args, nn_data: dai.NNData):
    """
    Custom decode function for the NN component.
    The blob now outputs 6 values per detection: [x, y, w, h, conf, class]
    """
    # Set expected number of output elements to 6 (removing orientation and depth)
    no = 6  # REMOVED: args.orientation and args.depth adjustments
    output = torch.tensor(nn_data.getLayerFp16("output0"))
    results = non_max_suppression(output.view(1, no, -1))
    dets = Detections(nn_data)
    r = results[0]
    
    if r.numel() > 0:
        for result in r:            
            x_min = result[0].item()
            y_min = result[1].item()
            x_max = result[2].item()
            y_max = result[3].item()
            conf = result[4].item()
            label = int(result[5].item())

            # REMOVED: depth and orientation extraction
            det = Detection(None, label, conf, x_min, y_min, x_max, y_max)
            dets.detections.append(det)

    return dets

# Constants and configurations
GPIO = dai.BoardConfig.GPIO
UART_NUM = 3
UART_CONFIGS = {
    0: {'txPin': 15, 'rxPin': 16, 'txPinMode': GPIO.ALT_MODE_2, 'rxPinMode': GPIO.ALT_MODE_2},
    2: {'txPin': 45, 'rxPin': 46, 'txPinMode': GPIO.ALT_MODE_3, 'rxPinMode': GPIO.ALT_MODE_3},
    3: {'txPin': 34, 'rxPin': 35, 'txPinMode': GPIO.ALT_MODE_5, 'rxPinMode': GPIO.ALT_MODE_5},
}

LABELS_1 = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
             "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

LABELS_2 = ["none", "car", "person"]

FRAME_RATE = 5
SIDE_FRAME_RATE = 5

def create_pipeline(args, main_model_config, side_model_config):
    pipeline = dai.Pipeline()
    
    # Create color camera
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
    cam_rgb.setIspScale(7,10)
    cam_rgb.setPreviewSize(2839, 2128)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(FRAME_RATE)

    # Create ImageManip for cropping and resizing
    manip = pipeline.create(dai.node.ImageManip)
    manip.setMaxOutputFrameSize(main_model_config['maxOutputFrameSize'])
    manip.initialConfig.setResizeThumbnail(main_model_config['nnShape'][0], main_model_config['nnShape'][1])
    manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
    if args.centerCrop:
        crop_factor = float(args.centerCrop)
        print(f"Crop factor: {crop_factor}")
        manip.initialConfig.setCenterCrop(crop_factor, main_model_config['nnShape'][0]/main_model_config['nnShape'][1])

    # Create neural network node
    if main_model_config.get('type') == 'yolo':
        detection_nn = pipeline.create(dai.node.NeuralNetwork)
        with open(main_model_config['configPath']) as f:
            config = json.load(f)
        LABELS = config['mappings']['labels']
    else:
        detection_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
        detection_nn.setConfidenceThreshold(0.5)
        LABELS = main_model_config.get('labels', LABELS_1)
    
    detection_nn.setBlobPath(main_model_config['blobPath'])
    detection_nn.input.setBlocking(False)

    # Linking
    cam_rgb.preview.link(manip.inputImage)
    manip.out.link(detection_nn.input)

    # Create outputs
    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("detections")
    detection_nn.out.link(xout_nn.input)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("preview")
    detection_nn.passthrough.link(xout_rgb.input)

    if args.blindSpotDetection:
        # Create left and right cameras for blind spot detection (unchanged)
        cam_left = pipeline.create(dai.node.MonoCamera)
        cam_right = pipeline.create(dai.node.MonoCamera)
        cam_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        cam_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        cam_left.setFps(SIDE_FRAME_RATE)
        cam_right.setFps(SIDE_FRAME_RATE)

        manip_left = pipeline.create(dai.node.ImageManip)
        manip_right = pipeline.create(dai.node.ImageManip)
        manip_left.initialConfig.setResize(side_model_config['nnShape'][0], side_model_config['nnShape'][1])
        manip_right.initialConfig.setResize(side_model_config['nnShape'][0], side_model_config['nnShape'][1])
        manip_left.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        manip_right.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        manip_left.setMaxOutputFrameSize(side_model_config['maxOutputFrameSize'])
        manip_right.setMaxOutputFrameSize(side_model_config['maxOutputFrameSize'])

        if side_model_config.get('type') == 'yolo':
            detection_nn_left = pipeline.create(dai.node.NeuralNetwork)
            detection_nn_right = pipeline.create(dai.node.NeuralNetwork)
            with open(side_model_config['configPath']) as f:
                config = json.load(f)
        else:
            detection_nn_left = pipeline.create(dai.node.MobileNetDetectionNetwork)
            detection_nn_right = pipeline.create(dai.node.MobileNetDetectionNetwork)
            detection_nn_left.setConfidenceThreshold(0.5)
            detection_nn_right.setConfidenceThreshold(0.5)
        
        detection_nn_left.setBlobPath(side_model_config['blobPath'])
        detection_nn_right.setBlobPath(side_model_config['blobPath'])
        detection_nn_left.input.setBlocking(False)
        detection_nn_right.input.setBlocking(False)

        cam_left.out.link(manip_left.inputImage)
        cam_right.out.link(manip_right.inputImage)
        manip_left.out.link(detection_nn_left.input)
        manip_right.out.link(detection_nn_right.input)

        xout_left = pipeline.create(dai.node.XLinkOut)
        xout_right = pipeline.create(dai.node.XLinkOut)
        xout_left.setStreamName("detections_left")
        xout_right.setStreamName("detections_right")
        detection_nn_left.out.link(xout_left.input)
        detection_nn_right.out.link(xout_right.input)

        xout_left_preview = pipeline.create(dai.node.XLinkOut)
        xout_right_preview = pipeline.create(dai.node.XLinkOut)
        xout_left_preview.setStreamName("preview_left")
        xout_right_preview.setStreamName("preview_right")
        cam_left.out.link(xout_left_preview.input)
        cam_right.out.link(xout_right_preview.input)

    return pipeline, main_model_config['nnShape'], side_model_config['nnShape']

def run_pipeline(pipeline, args, main_nn_shape, side_nn_shape):
    with dai.Device(pipeline) as device:
        preview_queue = device.getOutputQueue("preview", maxSize=4, blocking=False)
        detection_queue = device.getOutputQueue("detections", maxSize=4, blocking=False)
        start_time = time.monotonic()
        counter = 0
        fps = 0
        frame_interval = 1 / FRAME_RATE
        next_frame_time = start_time
        
        if args.blindSpotDetection:
            left_queue = device.getOutputQueue("detections_left", maxSize=4, blocking=False)
            right_queue = device.getOutputQueue("detections_right", maxSize=4, blocking=False)
            left_preview_queue = device.getOutputQueue("preview_left", maxSize=4, blocking=False)
            right_preview_queue = device.getOutputQueue("preview_right", maxSize=4, blocking=False)
            side_frame_interval = 1 / SIDE_FRAME_RATE
            next_side_frame_time = start_time

        while True:
            current_time = time.monotonic()
            
            if current_time >= next_frame_time:
                in_rgb = preview_queue.get()
                detections = detection_queue.tryGet()
                frame = in_rgb.getCvFrame()
                
                counter += 1
                if (current_time - start_time) > 1:
                    fps = counter / (current_time - start_time)
                    counter = 0
                    start_time = current_time

                # Draw detections on the frame
                if detections is not None:
                    det = decode(args, detections)
                    print(f"Detections: {len(det.detections)}")
                    for detection in det.detections:
                        bbox = [int(detection.xmin), int(detection.ymin), int(detection.xmax), int(detection.ymax)]
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                        # REMOVED: depth and orientation label handling; now showing class and confidence only.
                        label_text = f"Cls: {detection.label} Conf: {detection.confidence:.2f}"
                        cv2.putText(frame, label_text, (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0))
                        
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255))
                window_name = f"Preview - Crop: {args.centerCrop}" if args.centerCrop else "Preview - Full FOV"
                cv2.imshow(window_name, frame)
                next_frame_time += frame_interval

            if args.blindSpotDetection and current_time >= next_side_frame_time:
                left_detections = left_queue.tryGet()
                right_detections = right_queue.tryGet()
                left_preview = left_preview_queue.get()
                right_preview = right_preview_queue.get()

                left_frame = left_preview.getCvFrame()
                right_frame = right_preview.getCvFrame()

                if left_detections:
                    left_det = decode(args, left_detections)
                    for detection in left_det.detections:
                        bbox = [int(detection.xmin), int(detection.ymin), int(detection.xmax), int(detection.ymax)]
                        cv2.rectangle(left_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.imshow("Left Blind Spot", left_frame)

                if right_detections:
                    right_det = decode(args, right_detections)
                    for detection in right_det.detections:
                        bbox = [int(detection.xmin), int(detection.ymin), int(detection.xmax), int(detection.ymax)]
                        cv2.rectangle(right_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.imshow("Right Blind Spot", right_frame)

                next_side_frame_time += side_frame_interval

            if cv2.waitKey(1) == ord('q'):
                break

            time.sleep(max(0, next_frame_time - time.monotonic()))

def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

def flash_pipeline(pipeline):
    (f, bl) = dai.DeviceBootloader.getFirstAvailableDevice()
    bootloader = dai.DeviceBootloader(bl)
    def progress(p): return print(f'Flashing progress: {p*100:.1f}%')
    bootloader.flash(progress, pipeline)

def build_dap(pipeline):
    dai.DeviceBootloader.saveDepthaiApplicationPackage(
        pipeline,
        compress=True,
        applicationName='HawkeyeTailunitV1'
    )

def parse_arguments():
    parser = argparse.ArgumentParser(description="Hawkeye Tail Unit")
    parser.add_argument("-m", "--model", type=str, default="mobilenet-ssd",
                        choices=["mobilenet-ssd", "ssdlite_mobilenet_v2", "person-vehicle-bike-detection-2000",
                                 "person-vehicle-bike-detection-2001", "person-vehicle-bike-detection-2002",
                                 "person-vehicle-bike-detection-2004", "vehicle-detection-0200",
                                 "vehicle-detection-adas-0002", "yolo-v4-tiny-tf",
                                 "pedestrian-and-vehicle-detector-adas-0001", "yolo-custom-v1", "custom_yolo"],
                        help="Choose the model to use for main camera object detection")
    parser.add_argument("--side-model", type=str, default=None,
                        choices=["mobilenet-ssd", "ssdlite_mobilenet_v2", "person-vehicle-bike-detection-2000",
                                 "person-vehicle-bike-detection-2001", "person-vehicle-bike-detection-2002",
                                 "person-vehicle-bike-detection-2004", "vehicle-detection-0200",
                                 "vehicle-detection-adas-0002", "yolo-v4-tiny-tf",
                                 "pedestrian-and-vehicle-detector-adas-0001"],
                        help="Choose the model to use for side cameras object detection (if different from main camera)")
    parser.add_argument("-c", "--centerCrop", type=str, default="",
                        help="Center crop factor (e.g., '0.7' for 70% crop)")
    parser.add_argument("-s", "--standalone", action="store_true",
                        help="Flash the pipeline to the device for standalone mode")
    parser.add_argument("-b", "--build", action="store_true",
                        help="Build a Depth AI Application Package (DAP)")
    # REMOVED: Depth and orientation arguments
    # parser.add_argument("-d", "--depth", action="store_true", help="Run model with depth prediction")
    # parser.add_argument("-o", "--orientation", action="store_true", help="Run model with orientation prediction")
    parser.add_argument("--blindSpotDetection", action="store_true",
                        help="Enable blind spot detection using side cameras")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    main_model_config = get_model_config(args.model)
    side_model_config = get_model_config(args.side_model if args.side_model else args.model)
    
    pipeline, main_nn_shape, side_nn_shape = create_pipeline(args, main_model_config, side_model_config)

    # Add UART configuration
    pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_4)
    uart_config = UART_CONFIGS[UART_NUM]
    
    board_config = dai.BoardConfig()
    board_config.gpio[uart_config['txPin']] = GPIO(GPIO.OUTPUT, uart_config['txPinMode'])
    board_config.gpio[uart_config['rxPin']] = GPIO(GPIO.INPUT, uart_config['rxPinMode'])
    board_config.uart[UART_NUM] = dai.BoardConfig.UART()
    
    pipeline.setBoardConfig(board_config)

    if args.standalone:
        flash_pipeline(pipeline)
    elif args.build:
        build_dap(pipeline)
    else:
        run_pipeline(pipeline, args, main_nn_shape, side_nn_shape)

if __name__ == "__main__":
    main()