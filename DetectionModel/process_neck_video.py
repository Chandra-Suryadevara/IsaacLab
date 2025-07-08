# process_neck_video_no_fallback.py
import argparse
import cv2
import numpy as np
import os
import torch
from collections import deque
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger

class MedianSmoother:
    """A robust smoother that uses the median of the last N points to reject outliers."""
    def __init__(self, window_size=7):
        self.window_size = window_size
        self.centers = deque(maxlen=window_size)
        self.radii = deque(maxlen=window_size)
        self.last_known_position = None

    def update(self, center, radius):
        """Add a new point and return the median of the current window."""
        self.centers.append(center)
        self.radii.append(radius)
        
        median_center_x = np.median([p[0] for p in self.centers])
        median_center_y = np.median([p[1] for p in self.centers])
        median_radius = np.median(self.radii)
        
        self.last_known_position = ((int(median_center_x), int(median_center_y)), int(median_radius))
        return self.last_known_position

    def get_last_smoothed(self):
        """Return the last known smoothed position."""
        return self.last_known_position

def draw_circle_from_box(frame, instances, metadata, smoother):
    """Draw a circle based on a median filter of high-confidence detections of the neck."""
    result = frame.copy()
    
    # --- MODIFIED ---
    # The logic to redraw the last known position has been removed.
    # If no instances are found above the threshold, this function will now
    # simply return the original frame without any drawing.
    if len(instances) == 0:
        return result

    boxes = instances.pred_boxes.tensor.cpu().numpy()
    scores = instances.scores.cpu().numpy()

    circle_color = (0, 255, 0)
    bullseye_color = (0, 0, 255)

    best_idx = np.argmax(scores)
    box = boxes[best_idx]
    score = scores[best_idx]

    x1, y1, x2, y2 = box.astype(int)
    width = x2 - x1
    height = y2 - y1

    # Transformations for a neck-only model
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2 - int(height * 0.25) 
    radius = int(min(width, height) * 0.75 / 2)

    # Get smoothed values from the Median Smoother
    smoothed_center, smoothed_radius = smoother.update((center_x, center_y), radius)
    
    cv2.circle(result, smoothed_center, smoothed_radius, circle_color, 2)
    cv2.putText(result, "InjectionArea", (smoothed_center[0] - 40, smoothed_center[1] - smoothed_radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, circle_color, 2)
    
    bullseye_radius = max(5, smoothed_radius // 10)
    cv2.circle(result, smoothed_center, bullseye_radius, bullseye_color, -1)

    info_text = f"Confidence: {score:.2f}"
    cv2.putText(result, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    return result

def main():
    parser = argparse.ArgumentParser(description="Run median filter smoothing on neck detections.")
    parser.add_argument("--config-file", required=True, help="path to config file")
    parser.add_argument("--input", required=True, help="path to a single input video")
    parser.add_argument("--output", required=True, help="path to save the output video")
    parser.add_argument("--confidence-threshold", type=float, default=0.7,
                        help="Only consider detections above this confidence. Default is 0.7")
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER, help="Modify config options")
    args = parser.parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")
    metadata.thing_classes = ["neck"]
    predictor = DefaultPredictor(cfg)
    
    # The smoother is still useful for stabilizing the detections that ARE found
    smoother = MedianSmoother(window_size=7)

    cap = cv2.VideoCapture(args.input)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    print(f"Processing video with confidence threshold ({args.confidence_threshold}) and NO fallback drawing...")
    
    for _ in tqdm(range(total_frames), desc="Video Processing"):
        ret, frame = cap.read()
        if not ret: break
        
        predictions = predictor(frame)
        instances = predictions["instances"].to("cpu")
        
        visualized_output = draw_circle_from_box(frame, instances, metadata, smoother)
        out.write(visualized_output)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video processing finished. Output saved to {args.output}")

if __name__ == "__main__":
    main()
