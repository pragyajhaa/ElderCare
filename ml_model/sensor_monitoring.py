import cv2
import numpy as np
import time
from datetime import datetime
import serial  # Assuming you're using serial communication for sensors

class SensorData:
    def __init__(self):
        # Initialize serial communication with your sensors
        self.serial_port = serial.Serial('/dev/ttyUSB0', 9600)  # Adjust as needed
        self.fall_detected = False
        self.heart_rate = 0
        self.temperature = 0.0

    def read_sensors(self):
        # Read sensor data (assuming it's coming in a specific format)
        if self.serial_port.in_waiting > 0:
            data = self.serial_port.readline().decode('utf-8').strip()
            # Parse the data, assuming it's comma-separated: fall, heart_rate, temperature
            self.fall_detected, self.heart_rate, self.temperature = map(float, data.split(','))

    def check_health_risks(self):
        # Example health risk detection based on sensor data
        if self.heart_rate < 50 or self.heart_rate > 100:
            return "Abnormal Heart Rate Detected"
        if self.temperature < 35 or self.temperature > 38:
            return "Abnormal Temperature Detected"
        return "Normal Health"

class MovementDetection:
    def __init__(self):
        self.prev_frame = None
        self.immobility_start_time = None
        self.fall_start_time = None
        self.seizure_start_time = None
        self.immobility_threshold = 300  # seconds (e.g., 5 minutes)
        self.fall_time_window = 3  # seconds (time window to analyze posture)
        self.seizure_time_window = 5  # seconds (time window to analyze seizures)
        self.seizure_movement_threshold = 10  # Threshold for detecting repetitive movements

    def detect_movement(self, frame, sensor_data):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return "Monitoring"

        diff = cv2.absdiff(self.prev_frame, gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.prev_frame = gray

        if sensor_data.fall_detected:
            return "Fall Detected (Sensor)"

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 500:  # Movement detected
                movement_type = self.analyze_movement(largest_contour, sensor_data)
                return movement_type
            else:
                return self.check_immobility(sensor_data)
        else:
            return self.check_immobility(sensor_data)

    def analyze_movement(self, contour, sensor_data):
        x, y, w, h = cv2.boundingRect(contour)

        if h > w * 2:  # Standing posture
            self.fall_start_time = None
            return "Normal Activity"
        else:
            if self.fall_start_time is None:
                self.fall_start_time = time.time()
            elif time.time() - self.fall_start_time > self.fall_time_window:
                return "Fall Detected (Video)"
            return "Monitoring Posture"

        if len(contour) > self.seizure_movement_threshold:
            if self.seizure_start_time is None:
                self.seizure_start_time = time.time()
            elif time.time() - self.seizure_start_time > self.seizure_time_window:
                return "Seizure Detected"
            return "Repetitive Movement Detected"

        return "Normal Activity"

    def check_immobility(self, sensor_data):
        current_time = datetime.now()
        if current_time.hour >= 22 or current_time.hour < 6:
            return "Monitoring during Sleep Time"

        if self.immobility_start_time is None:
            self.immobility_start_time = time.time()
        elif time.time() - self.immobility_start_time > self.immobility_threshold:
            return "Sudden Immobility Detected"

        return "Normal Activity"

def main():
    cap = cv2.VideoCapture(0)
    detector = MovementDetection()
    sensor_data = SensorData()

    if not cap.isOpened():
        print("Error: Cannot access the webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame from webcam")
            break

        sensor_data.read_sensors()  # Read the latest data from sensors

        movement_message = detector.detect_movement(frame, sensor_data)
        health_message = sensor_data.check_health_risks()

        # Combine both messages and display on screen
        display_message = f"{movement_message} | {health_message}"
        cv2.putText(frame, display_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Elderly Care Monitoring', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
