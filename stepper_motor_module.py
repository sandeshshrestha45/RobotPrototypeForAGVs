import RPi.GPIO as GPIO
import time

class StepperMotor:
    def __init__(self, control_pins):
        self.control_pins = control_pins
        # define advanced sequence as shown in the manufacturer's datasheet
        self.halfstep_seq = [
                [1,0,0,0],
                [1,1,0,0],
                [0,1,0,0],
                [0,1,1,0],
                [0,0,1,0],
                [0,0,1,1],
                [0,0,0,1],
                [1,0,0,1]]

        GPIO.setmode(GPIO.BCM)
        for pin in self.control_pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)

    def rotate(self, steps, direction = 1):
        for _ in range(steps):
            for halfstep in range(8):
                for pin in range(4):
                    if direction == 1:
                        GPIO.output(self.control_pins[pin], self.halfstep_seq[halfstep][pin])
                    else:
                        GPIO.output(self.control_pins[pin], self.halfstep_seq[7 - halfstep][pin])

                time.sleep(0.001)

    def cleanup(self):
        GPIO.cleanup()

'''
motor = StepperMotor(control_pins = [5, 6, 23, 24])

try:
     # rotate motor clockwise
    motor.rotate(512, direction = 1)

    # small delay before changing direction
    time.sleep(1)

    # rotate motor anti-clockwise
    motor.rotate(512, direction = -1)

finally:
    motor.cleanup()
'''