from flask import Flask, render_template, Response, request
from motor_module import Motor
from camera import Camera
from object_detector import ObjectDetector
from red_line_detector import RedLineDetector
import cv2
import threading
from time import time

app = Flask(__name__)

# Initialize motor
motor = Motor(2, 3, 4, 17, 22, 27)

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
red_line_detector = RedLineDetector(focal_length, red_line_height, motor)
camera = Camera()

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