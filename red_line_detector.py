import cv2
import numpy as np
from time import time
import threading

class RedLineDetector:
    def __init__(self, focal_length, red_line_height, motor):
        self.focal_length = focal_length
        self.red_line_height = red_line_height
        self.motor = motor
        self.red_line_stopped = False  # Flag to track if the motor has been stopped due to red line detection
        self.red_line_detected_time = 0  # Record the time when the red line was detected

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

        return False

    def handle_red_line(self):
        if self.red_line_stopped:
            if time() - self.red_line_detected_time >= self.motor.normal_running_duration:
                self.red_line_stopped = False  # Reset flag to allow normal running
                print("Normal running period after red line detection elapsed.")

    def stop_motor_due_to_red_line(self):
        print("Red Line detected. Stopping motor for 20 seconds.")
        self.motor.stop(0.1)  # stop the motor
        self.red_line_stopped = True  # Set flag to indicate motor is stopped due to red line detection
        self.red_line_detected_time = time()  # Record the time when the motor was stopped
        # Schedule re-enablement of motor start after 20 seconds
        threading.Timer(self.motor.red_line_stop_duration, self.motor.enable_start).start()