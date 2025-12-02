# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
This is the V11 Validator used by the 2025 capstone team.

Check a model's accuracy on a test or val split of a dataset.

Usage:
    $ yolo mode=val model=yolo11n.pt data=coco8.yaml imgsz=640

Usage - formats:
    $ yolo mode=val model=yolo11n.pt                 # PyTorch
                          yolo11n.torchscript        # TorchScript
                          yolo11n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                          yolo11n_openvino_model     # OpenVINO
                          yolo11n.engine             # TensorRT
                          yolo11n.mlpackage          # CoreML (macOS-only)
                          yolo11n_saved_model        # TensorFlow SavedModel
                          yolo11n.pb                 # TensorFlow GraphDef
                          yolo11n.tflite             # TensorFlow Lite
                          yolo11n_edgetpu.tflite     # TensorFlow Edge TPU
                          yolo11n_paddle_model       # PaddlePaddle
                          yolo11n.mnn                # MNN
                          yolo11n_ncnn_model         # NCNN
                          yolo11n_imx_model          # Sony IMX
                          yolo11n_rknn_model         # Rockchip RKNN
"""

import json
import time
from pathlib import Path
import math
import os
import matplotlib.pyplot as plt

import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode


class BaseValidator:
    """
    A base class for creating validators.

    This class provides the foundation for validation processes, including model evaluation, metric computation, and
    result visualization.

    Attributes:
        args (SimpleNamespace): Configuration for the validator.
        dataloader (DataLoader): Dataloader to use for validation.
        pbar (tqdm): Progress bar to update during validation.
        model (nn.Module): Model to validate.
        data (dict): Data dictionary containing dataset information.
        device (torch.device): Device to use for validation.
        batch_i (int): Current batch index.
        training (bool): Whether the model is in training mode.
        names (dict): Class names mapping.
        seen (int): Number of images seen so far during validation.
        stats (dict): Statistics collected during validation.
        confusion_matrix: Confusion matrix for classification evaluation.
        nc (int): Number of classes.
        iouv (torch.Tensor): IoU thresholds from 0.50 to 0.95 in spaces of 0.05.
        jdict (list): List to store JSON validation results.
        speed (dict): Dictionary with keys 'preprocess', 'inference', 'loss', 'postprocess' and their respective
            batch processing times in milliseconds.
        save_dir (Path): Directory to save results.
        plots (dict): Dictionary to store plots for visualization.
        callbacks (dict): Dictionary to store various callback functions.

    Methods:
        __call__: Execute validation process, running inference on dataloader and computing performance metrics.
        match_predictions: Match predictions to ground truth objects using IoU.
        add_callback: Append the given callback to the specified event.
        run_callbacks: Run all callbacks associated with a specified event.
        get_dataloader: Get data loader from dataset path and batch size.
        build_dataset: Build dataset from image path.
        preprocess: Preprocess an input batch.
        postprocess: Postprocess the predictions.
        init_metrics: Initialize performance metrics for the YOLO model.
        update_metrics: Update metrics based on predictions and batch.
        finalize_metrics: Finalize and return all metrics.
        get_stats: Return statistics about the model's performance.
        check_stats: Check statistics.
        print_results: Print the results of the model's predictions.
        get_desc: Get description of the YOLO model.
        on_plot: Register plots (e.g. to be consumed in callbacks).
        plot_val_samples: Plot validation samples during training.
        plot_predictions: Plot YOLO model predictions on batch images.
        pred_to_json: Convert predictions to JSON format.
        eval_json: Evaluate and return JSON format of prediction statistics.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        Initialize a BaseValidator instance.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to be used for validation.
            save_dir (Path, optional): Directory to save results.
            pbar (tqdm.tqdm, optional): Progress bar for displaying progress.
            args (SimpleNamespace, optional): Configuration for the validator.
            _callbacks (dict, optional): Dictionary to store various callback functions.
        """
        self.args = get_cfg(overrides=args)
        self.dataloader = dataloader
        self.pbar = pbar
        self.stride = None
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = True
        self.names = None
        self.seen = None
        self.stats = None
        self.confusion_matrix = None
        self.nc = None
        self.iouv = None
        self.jdict = None
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}

        self.save_dir = save_dir or get_save_dir(self.args)
        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        if self.args.conf is None:
            self.args.conf = 0.001  # default conf=0.001
        self.args.imgsz = check_imgsz(self.args.imgsz, max_dim=1)

        self.plots = {}
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

    def generate_depth_metrics(self):
        """
        Generate metrics for depth data across different IoU thresholds and depth bounds.
        Creates boxplots showing depth error distributions by true depth value.
        """
        iou_thresholds = [round(threshold, 2) for threshold in self.iouv.cpu().tolist()]

        # Define depth bounds and corresponding labels
        bounds = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0]
        labels = ["0-4", "5-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99", "100-109", "110-119", "120-129"]

        
        for i, thresh in enumerate(iou_thresholds):
            LOGGER.info(f"Processing depth metrics for IoU threshold: {thresh}")
            depth_data = []
            angle_data = []
            
            #dep_path_box = self.save_dir / f"{thresh}_depth_boxplot_2.png"
            #ang_path_box = self.save_dir / f"{thresh}_angle_boxplot_2.png"
            
            # Collect data for each depth bound
            valid_depth_data_exists = False
            valid_angle_data_exists = False
            valid_labels = []
            valid_depth = []
            valid_angle = []
           
            for j, bound in enumerate(bounds):
                #depth_path = self.save_dir / f"{thresh}_{bound}_depth.txt"
                #angle_path = self.save_dir / f"{thresh}_{bound}_angle.txt"
                
                # Load depth data if file exists
                if os.path.exists(depth_path):
                    try:
                        dep_data = np.loadtxt(depth_path)
                        os.remove(depth_path)  # Clean up file after reading
                        
                        # Filter valid depth data (non-negative values)
                        dep_data = dep_data[dep_data >= 0]
                        
                        if len(dep_data) > 0:
                            LOGGER.info(f"Found {len(dep_data)} valid depth data points for bound {bound}")
                            valid_depth_data_exists = True
                            valid_labels.append(labels[j])
                            valid_depth.append(dep_data)
                        else:
                            LOGGER.info(f"No valid depth data for bound {bound} after filtering")
                    except Exception as e:
                        LOGGER.warning(f"Error loading depth data from {depth_path}: {e}")
                        dep_data = np.array([])
                else:
                    LOGGER.info(f"Depth file not found: {depth_path}")
                    dep_data = np.array([])
                
                depth_data.append(dep_data)
                
                # Load angle data if file exists
                if os.path.exists(angle_path):
                    try:
                        ang_data = np.loadtxt(angle_path)
                        os.remove(angle_path)  # Clean up file after reading
                        
                        # Filter valid angle data (values <= 180 degrees)
                        ang_data = ang_data[ang_data <= 180]
                        
                        if len(ang_data) > 0:
                            LOGGER.info(f"Found {len(ang_data)} valid angle data points for bound {bound}")
                            valid_angle_data_exists = True
                            valid_angle.append(ang_data)
                        else:
                            LOGGER.info(f"No valid angle data for bound {bound} after filtering")
                    except Exception as e:
                        LOGGER.warning(f"Error loading angle data from {angle_path}: {e}")
                        ang_data = np.array([])
                else:
                    LOGGER.info(f"Angle file not found: {angle_path}")
                    ang_data = np.array([])
                
                angle_data.append(ang_data)

        # Plot depth box plots if we have valid data
        """
        if valid_depth_data_exists:
            plt.figure(figsize=(len(valid_labels) * 1.5, 6))
            plt.boxplot(valid_depth, showfliers=False, labels=valid_labels, showmeans=True)
            plt.xlabel("True Depth Value (m)")
            plt.ylabel("Depth Error (m)")
            plt.title(f"Depth Error Distributions by True Depth Value (IoU={thresh})")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(dep_path_box, format="png")
            plt.close()
            LOGGER.info(f"Saved depth boxplot to {dep_path_box}")
        else:
            LOGGER.warning(f"No valid depth data to plot for IoU threshold {thresh}")

        # Plot angle box plots if we have valid data
        if valid_angle_data_exists:
            plt.figure(figsize=(len(valid_labels) * 1.5, 6))
            plt.boxplot(valid_angle, showfliers=False, labels=valid_labels, showmeans=True)
            plt.xlabel("True Depth Value (m)")
            plt.ylabel("Orientation Error (degrees)")
            plt.title(f"Orientation Error Distributions by True Depth Value (IoU={thresh})")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(ang_path_box, format="png")
            plt.close()
            LOGGER.info(f"Saved angle boxplot to {ang_path_box}")
        else:
            LOGGER.warning(f"No valid angle data to plot for IoU threshold {thresh}")
        """
        
    
    def generate_depth_metrics_fixed(self):
        """
        Generate comprehensive depth metrics visualizations grouped by depth ranges.
        Creates both boxplots showing error distributions and line plots showing trends 
        across different depth ranges.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        # Define depth bounds and corresponding labels
        bounds = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0]
        labels = ["0-4", "5-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99", "100-109", "110-119", "120-129"]
        
        # Get IoU thresholds
        iou_thresholds = [round(threshold, 2) for threshold in self.iouv.cpu().tolist()]
        
        # ---- Process depth data by range for each IoU threshold ----
        depth_data_by_threshold_and_range = {}
        
        # Check if we have depth data in the metric dictionary
        if not hasattr(self, 'metric_depth_data') or not self.metric_depth_data:
            LOGGER.info("No metric_depth_data available. Make sure it's being populated during detection.")
            return
        
        # First, organize data by depth range
        for thresh, depth_entries in self.metric_depth_data.items():
            if not isinstance(depth_entries, list) or not depth_entries:
                continue
                
            # Initialize data structure for this threshold
            depth_data_by_threshold_and_range[thresh] = [[] for _ in range(len(bounds)-1)]
            
            # Group data by depth range
            for entry in depth_entries:
                # Check that we have both true depth and the error value
                if 'depth' not in entry or 'error' not in entry:
                    continue
                    
                true_depth = entry['depth']
                error = entry['error']  # This should be the actual error, not the depth value
                
                # Find appropriate bin for this depth
                for i in range(len(bounds)-1):
                    if bounds[i] <= true_depth < bounds[i+1]:
                        depth_data_by_threshold_and_range[thresh][i].append(error)
                        break
        
        # ---- Create individual boxplots for each IoU threshold ----
        for thresh, depth_ranges in depth_data_by_threshold_and_range.items():
            # Create output paths for this threshold
            dep_path_box = self.save_dir / f"depth_error_by_range_iou_{thresh}.png"
            
            # Get non-empty ranges
            valid_data = []
            valid_labels = []
            
            for i, errors in enumerate(depth_ranges):
                if errors:  # If we have data for this range
                    valid_data.append(errors)
                    valid_labels.append(labels[i])
            
            if not valid_data:
                LOGGER.info(f"No valid depth data for threshold {thresh}, skipping boxplot.")
                continue
            
            # Create boxplot with fixed styling
            plt.figure(figsize=(15, 7))
            
            box = plt.boxplot(valid_data, 
                    labels=valid_labels, 
                    showfliers=False,
                    showmeans=True,
                    patch_artist=True,  # Enable patch_artist to use facecolor
                    meanprops={'marker': '^', 'markerfacecolor': 'green', 'markeredgecolor': 'green', 'markersize': 8})
            
            # Apply colors to boxes after creation
            for patch in box['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_edgecolor('blue')
            
            # Set other properties
            for median in box['medians']:
                median.set_color('red')
                median.set_linewidth(1.5)
            
            for whisker in box['whiskers']:
                whisker.set_color('black')
                whisker.set_linewidth(1.5)
                
            for cap in box['caps']:
                cap.set_color('black')
                cap.set_linewidth(1.5)
            
            # Add mean values as text
            means = [np.mean(data) if data else 0 for data in valid_data]
            for i, mean in enumerate(means):
                plt.text(i + 1, mean, f"{mean:.2f}", ha='center', va='bottom', color='green', fontweight='bold')
            
            # Add a horizontal line at y=0 to indicate zero error
            plt.axhline(y=0, color='green', linestyle='-', alpha=0.5)
            
            plt.xlabel("True Depth Range (m)", fontsize=12, fontweight='bold')
            plt.ylabel("Depth Error (m)", fontsize=12, fontweight='bold')
            plt.title(f"Depth Error Distributions by True Depth Range (IoU={thresh})", fontsize=14, fontweight='bold')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(dep_path_box, format="png", dpi=300)
            plt.close()
            
            LOGGER.info(f"Created depth error boxplot for IoU={thresh}")
        
        # ---- Create combined line plot ----
        if depth_data_by_threshold_and_range:
            plt.figure(figsize=(15, 8))
            
            line_styles = ['-', '--', '-.', ':']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            has_plotted_data = False
            
            for idx, (thresh, depth_ranges) in enumerate(depth_data_by_threshold_and_range.items()):
                # Calculate mean error for each depth range
                mean_errors = []
                valid_depth_ranges = []
                
                for i, errors in enumerate(depth_ranges):
                    if errors:
                        mean_errors.append(np.mean(errors))
                        # Use midpoint of range for x-axis
                        mid_point = (bounds[i] + bounds[i+1]) / 2
                        valid_depth_ranges.append(mid_point)
                
                if valid_depth_ranges:
                    has_plotted_data = True
                    line_style = line_styles[idx % len(line_styles)]
                    color = colors[idx % len(colors)]
                    plt.plot(valid_depth_ranges, mean_errors, 
                            marker='o', 
                            linestyle=line_style,
                            color=color,
                            linewidth=2.5, 
                            markersize=8,
                            label=f"IoU={thresh}")
            
            # Add a horizontal line at y=0 to indicate zero error
            plt.axhline(y=0, color='green', linestyle='-', alpha=0.5)
            
            plt.xlabel("Depth (m)", fontsize=12, fontweight='bold')
            plt.ylabel("Mean Depth Error (m)", fontsize=12, fontweight='bold')
            plt.title("Mean Depth Error by Distance for Different IoU Thresholds", fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            # Only add legend if we actually plotted some data
            if has_plotted_data and plt.gca().get_legend_handles_labels()[0]:
                plt.legend(fontsize=10)
                
            plt.tight_layout()
            plt.savefig(self.save_dir / "combined_depth_error_by_range.png", format="png", dpi=300)
            plt.close()
            
            LOGGER.info(f"Created combined depth error plot")
        
        # ---- Create depth error heatmap by ranges ----
        # Define depth thresholds (error values we consider acceptable)
        depth_thresholds = [15, 10, 5, 3, 2, 1]
        
        # Create a matrix to store percentages for heatmap
        if depth_data_by_threshold_and_range:
            dep_percentages = np.zeros((len(depth_thresholds), len(bounds)-1))
            
            # Choose the middle IoU threshold for the heatmap (or first if only one)
            representative_thresh = list(depth_data_by_threshold_and_range.keys())[len(depth_data_by_threshold_and_range)//2 
                                                                            if len(depth_data_by_threshold_and_range) > 1 else 0]
            depth_ranges = depth_data_by_threshold_and_range[representative_thresh]
            
            # Calculate percentages for each depth range and threshold
            for i, errors in enumerate(depth_ranges):
                if not errors:
                    dep_percentages[:, i] = -1  # No data
                    continue
                    
                total = len(errors)
                for j, threshold in enumerate(depth_thresholds):
                    # Count errors below the threshold (absolute value)
                    correct = sum(abs(error) <= threshold for error in errors)
                    dep_percentages[j, i] = (correct / total * 100) if total > 0 else -1
            
            # Remove columns with no data
            valid_columns = []
            valid_labels = []
            for i in range(dep_percentages.shape[1]):
                if not all(dep_percentages[:, i] == -1):
                    valid_columns.append(i)
                    valid_labels.append(labels[i])
            
            if valid_columns:
                dep_percentages = dep_percentages[:, valid_columns]
                
                # Create heatmap
                plt.figure(figsize=(len(valid_labels) * 1.5, len(depth_thresholds) * 1.5))
                heatmap = plt.imshow(dep_percentages, cmap="viridis", aspect="auto")
                plt.colorbar(label="Percent Correct")
                plt.xticks(ticks=np.arange(len(valid_labels)), labels=valid_labels, rotation=45)
                plt.yticks(ticks=np.arange(len(depth_thresholds)), labels=depth_thresholds)
                plt.xlabel("True Depth Range (m)")
                plt.ylabel("Depth Error Threshold (m)")
                plt.title(f"Depth Accuracy by Range (IoU={representative_thresh})")
                
                # Add text annotations
                for j in range(len(depth_thresholds)):
                    for i in range(len(valid_columns)):
                        value = dep_percentages[j, i]
                        if value >= 0:  # Only show text for valid data
                            plt.text(i, j, f"{value:.1f}%", 
                                    ha="center", va="center", 
                                    color="black" if value > 80 else "white")
                
                plt.tight_layout()
                plt.savefig(self.save_dir / "depth_accuracy_heatmap_by_range.png", format="png", dpi=300)
                plt.close()
                
                LOGGER.info(f"Created depth accuracy heatmap by depth range")
        
        print(f"Generated depth metrics visualizations focused on depth ranges")
        print(f"Output saved to {self.save_dir}")

    def calculate_angle_percentage(self):
        iou_thresholds = [round(threshold, 2) for threshold in self.iouv.cpu().tolist()]
        degree_thresholds = [180 / (j * 6) for j in range(1, 7)]
        percentages = np.zeros((len(degree_thresholds), len(iou_thresholds)))
        percent_path = self.save_dir / "angle_percent_results.png" 
        for i, threshold in enumerate(iou_thresholds):
                #angle_path = self.save_dir / f"{round(threshold, 2)}_results.txt"
                #data = np.loadtxt(angle_path)
                #os.remove(angle_path)

                data = np.array(self.metric_angle_data.get(round(threshold,2), []))  # <-- FIX: use int key

                tot = torch.sum(torch.from_numpy(data) >= 0)
                if tot == 0:
                    percentages[j, i] = -1
                else:
                    for j, threshold in enumerate(degree_thresholds):
                        correct_predictions = np.sum((data <= threshold) & (data >= 0))
                        percentages[j, i] = correct_predictions / tot * 100

        plt.figure(figsize=(len(iou_thresholds) * 1.5, len(degree_thresholds) * 1.5))
        heatmap = plt.imshow(percentages, cmap="viridis", aspect="auto")
        plt.colorbar(label="Percent Correct")
        plt.xticks(ticks=np.arange(len(iou_thresholds)), labels=iou_thresholds, rotation=45)
        plt.yticks(ticks=np.arange(len(degree_thresholds)), labels=np.round(degree_thresholds, 2))
        plt.xlabel("IoU Threshold")
        plt.ylabel("Degree Threshold")
        plt.title("Angle Match Percentages")
        for j in range(len(degree_thresholds)):
            for i in range(len(iou_thresholds)):
                plt.text(i, j, f"{percentages[j, i]:.2f}%", ha="center", va="center", color="black" if percentages[j, i] > 80 else "white")
        plt.tight_layout()
        LOGGER.info(f"Saving figure to {percent_path}, percentages shape: {percentages.shape}")
        plt.savefig(percent_path, format="png")
        plt.close()

    # Generate heatmaps and box-plots for angle and depth accuracy metrics
    def generate_metrics(self):
        iou_thresholds = [round(threshold, 2) for threshold in self.iouv.cpu().tolist()]
        degree_thresholds = [180 / (j * 6) for j in range(1, 7)]
        depth_thresholds = [15, 10, 5, 3, 2, 1]
        ang_percentages = np.zeros((len(degree_thresholds), len(iou_thresholds)))
        dep_percentages = np.zeros((len(depth_thresholds), len(iou_thresholds)))
        ang_path = self.save_dir / "angle_percent.png"
        ang_path_box = self.save_dir / "angle_boxplot.png"
        dep_path = self.save_dir / "depth_percent.png"
        dep_path_box = self.save_dir / "depth_boxplot.png"
        degree_data = []
        depth_data = []
        for i, thresh in enumerate(iou_thresholds):
                angle_path = self.save_dir / f"{round(thresh, 2)}_angle.txt"
                depth_path = self.save_dir / f"{round(thresh, 2)}_depth.txt"

                #ang_data = np.loadtxt(angle_path) if os.path.exists(angle_path) else None
                ang_data = np.array(self.metric_angle_data.get(round(thresh,2), [])) 

                #dep_data = np.loadtxt(depth_path) if os.path.exists(depth_path) else None
                #dep_data = np.array(self.metric_depth_data.get(round(thresh,2), [])) 
                dep_entries = self.metric_depth_data.get(round(thresh,2), [])
                dep_data = np.array([entry['error'] for entry in dep_entries])

                ang_tot = 0
                dep_tot = 0

                if ang_data is not None:
                    #os.remove(angle_path)
                    ang_tot = np.sum(ang_data <= 180)
                    degree_data.append(ang_data[ang_data <= 180])
                else:
                    degree_data.append([])

                if dep_data is not None:
                    #os.remove(depth_path)
                    dep_data = dep_data[dep_data >= 0]
                    dep_tot = len(dep_data)
                    depth_data.append(dep_data)
                else:
                    depth_data.append([])

                if ang_tot == 0:
                    # Initialize all thresholds to -1 for this IoU threshold
                    for j in range(len(degree_thresholds)):
                        ang_percentages[j, i] = -1
                else:
                    for j, threshold in enumerate(degree_thresholds):
                        correct_predictions = np.sum(ang_data <= threshold)
                        ang_percentages[j, i] = correct_predictions / ang_tot * 100

                if dep_tot == 0:
                    # Initialize all thresholds to -1 for this IoU threshold
                    for j in range(len(depth_thresholds)):
                        dep_percentages[j, i] = -1
                else:
                    for j, threshold in enumerate(depth_thresholds):
                        correct_predictions = np.sum(dep_data <= threshold)
                        dep_percentages[j, i] = correct_predictions / dep_tot * 100

        # Plot degree heatmap
        plt.figure(figsize=(len(iou_thresholds) * 1.5, len(degree_thresholds) * 1.5))
        heatmap = plt.imshow(ang_percentages, cmap="viridis", aspect="auto")
        plt.colorbar(label="Percent Correct")
        plt.xticks(ticks=np.arange(len(iou_thresholds)), labels=iou_thresholds, rotation=45)
        plt.yticks(ticks=np.arange(len(degree_thresholds)), labels=np.round(degree_thresholds, 2))
        plt.xlabel("IoU Threshold")
        plt.ylabel("Degree Threshold")
        plt.title("Angle Match Percentages")
        for j in range(len(degree_thresholds)):
            for i in range(len(iou_thresholds)):
                plt.text(i, j, f"{ang_percentages[j, i]:.2f}%", ha="center", va="center", color="black" if ang_percentages[j, i] > 80 else "white")
        plt.tight_layout()
        plt.savefig(ang_path, format="png")
        plt.close()

        # Plot depth heatmap
        plt.figure(figsize=(len(iou_thresholds) * 1.5, len(depth_thresholds) * 1.5))
        heatmap = plt.imshow(dep_percentages, cmap="viridis", aspect="auto")
        plt.colorbar(label="Percent Correct")
        plt.xticks(ticks=np.arange(len(iou_thresholds)), labels=iou_thresholds, rotation=45)
        plt.yticks(ticks=np.arange(len(depth_thresholds)), labels=np.round(depth_thresholds, 2))
        plt.xlabel("IoU Threshold")
        plt.ylabel("Distance Threshold (m)")
        plt.title("Depth Match Percentages")
        for j in range(len(depth_thresholds)):
            for i in range(len(iou_thresholds)):
                plt.text(i, j, f"{dep_percentages[j, i]:.2f}%", ha="center", va="center", color="black" if dep_percentages[j, i] > 80 else "white")
        plt.tight_layout()
        plt.savefig(dep_path, format="png")
        plt.close()

        # Plot degree box plots
        plt.figure(figsize=(len(iou_thresholds) * 1.5, 6))
        plt.boxplot(degree_data, showfliers=False, labels=iou_thresholds, showmeans=True)
        plt.xlabel("IoU Threshold")
        plt.ylabel("Degree Error")
        plt.title("Degree Error Distributions by IoU Threshold")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(ang_path_box, format="png")
        plt.close()

        # Plot depth box plots
        plt.figure(figsize=(len(iou_thresholds) * 1.5, 6))
        plt.boxplot(depth_data, showfliers=False, labels=iou_thresholds, showmeans=True, widths=.5)
        plt.xlabel("IoU Threshold")
        plt.ylabel("Depth Error (m)")
        plt.title("Depth Error Distributions by IoU Threshold")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(dep_path_box, format="png")
        plt.close()


    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """
        Execute validation process, running inference on dataloader and computing performance metrics.

        Args:
            trainer (object, optional): Trainer object that contains the model to validate.
            model (nn.Module, optional): Model to validate if not using a trainer.

        Returns:
            stats (dict): Dictionary containing validation statistics.
        """
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            # Force FP16 val during training
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            # self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("WARNING ⚠️ validating an untrained model YAML will result in 0 mAP.")
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            # self.model = model
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = model.metadata.get("batch", 1)  # export.py models default to batch-size 1
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not pt:
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # warmup

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                preds = model(batch["img"], augment=augment)

            # Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]

            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)
            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        print("Using V11 Ultralytics")
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w", encoding="utf-8") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            
            self.calculate_angle_percentage()
            self.generate_depth_metrics_fixed()
            self.generate_metrics()

            return stats

    def box_iou(box1, box2):
        """
        box1: [N, 4]
        box2: [M, 4]
        Format: (x1, y1, x2, y2)
        Returns: IoU [N, M]
        """
        def area(box):
            return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

        area1 = area(box1)
        area2 = area(box2)

        lt = torch.max(box1[:, None, :2], box2[:, :2])  # top-left
        rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # bottom-right

        wh = (rb - lt).clamp(min=0)  # width-height
        inter = wh[:, :, 0] * wh[:, :, 1]

        union = area1[:, None] + area2 - inter
        return inter / union

    def match_predictions(self, pred_classes, true_classes, pred_angle, true_angle, pred_depth, true_depth, iou, use_scipy=False):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        correct_angle = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)        
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class # zero out the wrong classes
        iou = iou.cpu().numpy()
        angle_diff = torch.abs(true_angle[:, None] - pred_angle).squeeze(1)
        diff_mask = angle_diff > 6.3
        angle_diff = torch.cos(angle_diff)
        angle_diff = torch.acos(angle_diff) * 180 / math.pi
        angle_diff[diff_mask] = -1
        angle_diff = angle_diff.cpu().numpy()
       
        # Initialize storage 
        if not hasattr(self, 'metric_angle_data'):
            self.metric_angle_data = {round(t,2): [] for t in self.iouv.cpu().tolist()}
        # Initialize storage 
        if not hasattr(self, 'metric_depth_data'):
            self.metric_depth_data = {round(t,2): [] for t in self.iouv.cpu().tolist()}
  
        
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            angle_path = self.save_dir / f"{round(threshold, 2)}_results.txt"
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
                if not self.training:
                    for match in matches:
                        tar_idx = match[0]
                        pred_idx = match[1]
                        
                        #with open(angle_path, "a") as file:
                        #    file.write(f"{angle_diff[tar_idx][pred_idx]}\n")

                        
                        self.metric_angle_data[round(threshold,2)].append(angle_diff[tar_idx][pred_idx]) #Store in  memory instead of file
                      
                        depth_error = abs(pred_depth[pred_idx] - true_depth[tar_idx])  # Compute error
                        true_depth_value = true_depth[tar_idx].item()  # Save the actual depth too
                        self.metric_depth_data[round(threshold,2)].append({
                            'depth': true_depth_value,
                            'error': depth_error.item()
                        })

                      

        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def add_callback(self, event: str, callback):
        """Append the given callback to the specified event."""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """Run all callbacks associated with a specified event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def get_dataloader(self, dataset_path, batch_size):
        """Get data loader from dataset path and batch size."""
        raise NotImplementedError("get_dataloader function not implemented for this validator")

    def build_dataset(self, img_path):
        """Build dataset from image path."""
        raise NotImplementedError("build_dataset function not implemented in validator")

    def preprocess(self, batch):
        """Preprocess an input batch."""
        return batch

    def postprocess(self, preds):
        """Postprocess the predictions."""
        return preds

    def init_metrics(self, model):
        """Initialize performance metrics for the YOLO model."""
        pass

    def update_metrics(self, preds, batch):
        """Updates metrics based on predictions and batch."""
        pass
        """if len(preds) and len(batch['bboxes']):
        # Assume your preds are a list of N predictions per image (e.g., per batch item)
        # You may need to modify this based on how your `postprocess` function structures `preds`
            for i in range(len(preds)):
                pred_boxes = preds[i]['boxes']
                pred_classes = preds[i]['cls']
                pred_angles = preds[i]['angle']  # Assuming you have 'angle' in prediction dict
                
                true_boxes = batch['bboxes'][i]
                true_classes = batch['cls'][i]
                true_angles = batch['angle'][i]
                
                # Compute IoU between predicted and ground truth boxes
                iou = self.box_iou(pred_boxes, true_boxes)  # You need a `box_iou` function
                
                self.match_predictions(
                    pred_classes=pred_classes,
                    true_classes=true_classes,
                    pred_angle=pred_angles,
                    true_angle=true_angles,
                    iou=iou,
                    use_scipy=True
                )"""

    def finalize_metrics(self, *args, **kwargs):
        """Finalize and return all metrics."""
        pass

    def get_stats(self):
        """Return statistics about the model's performance."""
        return {}

    def check_stats(self, stats):
        """Check statistics."""
        pass

    def print_results(self):
        """Print the results of the model's predictions."""
        pass

    def get_desc(self):
        """Get description of the YOLO model."""
        pass

    @property
    def metric_keys(self):
        """Return the metric keys used in YOLO training/validation."""
        return []

    def on_plot(self, name, data=None):
        """Register plots (e.g. to be consumed in callbacks)."""
        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}

    # TODO: may need to put these following functions into callback
    def plot_val_samples(self, batch, ni):
        """Plot validation samples during training."""
        pass

    def plot_predictions(self, batch, preds, ni):
        """Plot YOLO model predictions on batch images."""
        pass

    def pred_to_json(self, preds, batch):
        """Convert predictions to JSON format."""
        pass

    def eval_json(self, stats):
        """Evaluate and return JSON format of prediction statistics."""
        pass
