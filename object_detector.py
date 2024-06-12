import os
import cv2
import numpy as np
from cvlib.object_detection import YOLO
from time import time

class ObjectDetector:
    def __init__(self, weights, config, labels_path, target_labels, real_widths, focal_length):
        self.yolo = YOLO(weights, config, labels_path)
        self.target_labels = target_labels
        self.real_widths = real_widths
        self.focal_length = focal_length
        self.save_dir = "Detected_images"
        os.makedirs(self.save_dir, exist_ok=True)
        self.object_stopped = False  # Flag to track if the motor has been stopped due to object detection
        self.object_detected_time = 0  # Record the time when an object was detected

    def detect_objects(self, frame):
        bbox, labels, confidences = self.yolo.detect_objects(frame)
        if labels:  # Check if any class is detected
            self.object_detected_time = time()  # Record the time when an object is detected
            self.object_stopped = True  # Set flag to indicate motor is stopped due to object detection
        return bbox, labels, confidences

    def handle_object_detection(self):
        if self.object_stopped:
            # If motor is stopped due to object detection, check if object is still being detected for more than 0.5 seconds
            if time() - self.object_detected_time > 0.5:
                self.object_stopped = False  # Reset flag to allow normal running

    def draw_bbox(self, frame, bbox, labels, confidences):
        for box, label, conf in zip(bbox, labels, confidences):
            x, y, w, h = box
            color = (0, 255, 0)  # green color for the bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{label}: {conf:.2f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def calculate_distance_by_width(self, perceived_width, label):
        real_width = self.real_widths.get(label, 0)
        if perceived_width == 0:
            return float('inf')
        return (self.focal_length * real_width) / perceived_width