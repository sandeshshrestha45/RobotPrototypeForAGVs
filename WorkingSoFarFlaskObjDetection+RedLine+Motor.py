# app.py

from flask import Flask, render_template, Response, request
import cv2
import os
import numpy as np
from cvlib.object_detection import YOLO
import RPi.GPIO as GPIO
from time import time
import threading
from MotorModule2 import Motor

app = Flask(__name__)

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

class RedLineDetector:
    def __init__(self, focal_length, red_line_height):
        self.focal_length = focal_length
        self.red_line_height = red_line_height
        self.red_line_stopped = False # Flag to track if the motor has been stopped due to red line detection
        self.red_line_detected_time = 0 # Record the time when the red line was detected

    def detect_red_line(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_bound = np.array([160, 50, 50])
        upper_bound = np.array([180, 255, 255])

        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= 700:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                distance = (self.focal_length * self.red_line_height) / h
                print(f"Distance to red line: {distance} meters")

                if distance < 1.0:
                    return True

        red_line_detected = False # Placeholder for red line detection result

        return red_line_detected

    def handle_red_line(self):
        if self.red_line_stopped:
            if time() - self.red_line_detected_time >= motor.normal_running_duration:
                self.red_line_stopped = False # Reset flag to allow normal running
                print("Normal running period after red line detection elapsed.")


    def stop_motor_due_to_red_line(self):
        print("Red Line detected. Stopping motor for 20 seconds.")
        motor.stop(0.1) # stop the motor
        self.red_line_stopped = True # Set flag to indicate motor is stopped due to red line detection
        self.red_line_detected_time = time() # Record the time when the motor was stopped
        # Schedule re-enablement of motor start after 20 seconds
        threading.Timer(motor.red_line_stop_duration, motor.enable_start).start()

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # 0 for the first camera, 1 for the second, and so on
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"FPS set to: {self.fps}")

    def read_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()

    def close_windows(self):
        cv2.destroyAllWindows()

# Initialize object detector, red line detector, and camera
weights = "yolov4-tiny-obj_best.weights"
config = "yolov4-tiny-custom.cfg"
labels_path = "obj.names"
target_labels = ['Human', 'Forklift', 'Cone', 'Pallet']
real_widths = {
    'Human': 0.3,
    'Forklift': 1.5,
    'Cone': 0.3,
}
focal_length = 579.217877094
red_line_height = 0.175

detector = ObjectDetector(weights, config, labels_path, target_labels, real_widths, focal_length)
red_line_detector = RedLineDetector(focal_length, red_line_height)
camera = Camera()

# Initialize motor
motor = Motor(2, 3, 4, 17, 22, 27)

# Inside gen_frames() function
def gen_frames():
    while True:
        ret, frame = camera.read_frame()
        if not ret:
            break
        else:
            # Perform red line detection on the frame
            red_line_detected = red_line_detector.detect_red_line(frame)

            # Handle the red line detection period
            red_line_detector.handle_red_line()

            # If red line is detected and motor is not already stopped due to red line detection, stop the motor for 20 seconds
            if red_line_detected and not red_line_detector.red_line_stopped:
                red_line_detector.stop_motor_due_to_red_line()
            else:
                # Continue with object detection and processing if red line is not detected or if motor is already stopped due to red line detection
                bbox, labels, confs = detector.detect_objects(frame)

                for box, label, conf in zip(bbox, labels, confs):
                    distance = detector.calculate_distance_by_width(box[2], label)
                    print(f"Detected {label} at distance: {distance:.2f} meters")
                    detector.draw_bbox(frame, [box], [label], [conf])
                    # Override manual control if object detected within 0.4m
                    if distance < 0.4:
                        print("Stopping motor due to object detection within 0.4 meters")
                        motor.stop(0.1)
                        break  # Exit the loop to ensure motor stops immediately

                # Handle object detection period
                detector.handle_object_detection()

                # Encode the frame in JPEG format
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Yield the frame in byte format

@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML template

@app.route('/control', methods=['POST'])
def control():
    direction = request.form['direction']
    if direction == 'up':
        motor.move(speed=0.3)
    elif direction == 'down':
        motor.move(speed=-0.3)
    elif direction == 'left':
        motor.move(turn=-0.5)
    elif direction == 'right':
        motor.move(turn=0.5)
    elif direction == 'stop':
        motor.stop(0.1)
    return 'OK'

@app.route('/stop', methods=['POST'])
def stop():
    motor.stop(0.1)
    return 'OK'

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', threaded=True)