import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime

class FallDetection:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.prev_com_y = None  # Previous center of mass y-coordinate
        self.prev_time = None  # Time when the previous frame was processed
        self.fall_start_time = None
        self.immobility_start_time = None
        self.immobility_threshold = 5  # seconds (e.g., 5 seconds of immobility)
        self.vertical_threshold = 100  # Pixels: Change in COM to consider a fall
        self.velocity_threshold = 50  # Pixels/sec: Speed threshold for fall detection
        self.angle_threshold = 45  # Angle threshold for fall detection (degrees)
        self.landmarks = None

    def detect_movement(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            self.landmarks = results.pose_landmarks.landmark
            # Draw key landmarks and connections
            self.draw_key_landmarks(frame)
            
            movement_type = self.analyze_movement()
            return movement_type
        else:
            return self.check_immobility()

    def draw_key_landmarks(self, frame):
        key_landmarks = [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_EYE,
            self.mp_pose.PoseLandmark.RIGHT_EYE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE
        ]

        points = []
        for landmark_id in key_landmarks:
            landmark = self.landmarks[landmark_id]
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        if len(points) >= 15:
            # Body connections
            cv2.line(frame, points[3], points[9], (0, 255, 0), 2)
            cv2.line(frame, points[4], points[10], (0, 255, 0), 2)
            cv2.line(frame, points[9], points[11], (0, 255, 0), 2)
            cv2.line(frame, points[10], points[12], (0, 255, 0), 2)
            cv2.line(frame, points[11], points[13], (0, 255, 0), 2)
            cv2.line(frame, points[12], points[14], (0, 255, 0), 2)
            cv2.line(frame, points[3], points[4], (0, 255, 0), 2)
            cv2.line(frame, points[9], points[10], (0, 255, 0), 2)
            
            # Arm connections
            cv2.line(frame, points[3], points[5], (0, 255, 0), 2)
            cv2.line(frame, points[4], points[6], (0, 255, 0), 2)
            cv2.line(frame, points[5], points[7], (0, 255, 0), 2)
            cv2.line(frame, points[6], points[8], (0, 255, 0), 2)

            # Face connections
            cv2.line(frame, points[0], points[1], (0, 255, 0), 2)
            cv2.line(frame, points[0], points[2], (0, 255, 0), 2)

    def analyze_movement(self):
        if self.landmarks:
            # Calculate center of mass (COM) as average of hips
            hip_left = self.landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            hip_right = self.landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            com_x = (hip_left.x + hip_right.x) / 2
            com_y = (hip_left.y + hip_right.y) / 2
            
            # Calculate the distance between head and hips (height measure)
            head = self.landmarks[self.mp_pose.PoseLandmark.NOSE]
            head_hip_distance = abs(head.y - com_y)

            # Calculate time delta
            current_time = time.time()
            if self.prev_time is not None:
                time_delta = current_time - self.prev_time
                
                # Calculate vertical speed
                vertical_speed = abs(com_y - self.prev_com_y) / time_delta
                
                # Fall detection logic
                if vertical_speed > self.velocity_threshold and head_hip_distance < 0.2:  # Additional check: if the head is near the hips (collapsed posture)
                    # Check body orientation
                    shoulder_left = self.landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                    shoulder_right = self.landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    angle = self.calculate_angle(shoulder_left, shoulder_right, hip_left, hip_right)
                    
                    if angle < self.angle_threshold:
                        if self.fall_start_time is None:
                            self.fall_start_time = current_time
                        elif current_time - self.fall_start_time > 1:  # Stable detection
                            self.fall_start_time = None
                            return "Fall Detected", True
                        return "Monitoring Posture...", False
                else:
                    self.fall_start_time = None
            
            # Update previous COM and time
            self.prev_com_y = com_y
            self.prev_time = current_time
            return "Normal Activity", False
        else:
            return self.check_immobility()


    def calculate_angle(self, shoulder_left, shoulder_right, hip_left, hip_right):
        # Calculate angle of the line connecting shoulders and hips with respect to horizontal
        shoulder_mid_x = (shoulder_left.x + shoulder_right.x) / 2
        shoulder_mid_y = (shoulder_left.y + shoulder_right.y) / 2
        hip_mid_x = (hip_left.x + hip_right.x) / 2
        hip_mid_y = (hip_left.y + hip_right.y) / 2
        
        angle = np.degrees(np.arctan2(hip_mid_y - shoulder_mid_y, hip_mid_x - shoulder_mid_x))
        return abs(angle)
    
    def check_immobility(self):
        current_time = datetime.now()
        if current_time.hour >= 22 or current_time.hour < 6:
            return "Monitoring during Sleep Time", False

        if self.immobility_start_time is None:
            self.immobility_start_time = time.time()
        elif time.time() - self.immobility_start_time > self.immobility_threshold:
            return "Fall Detected", False

        return "Normal Activity", False

def main():
    cap = cv2.VideoCapture(0)
    detector = FallDetection()

    if not cap.isOpened():
        print("Error: Cannot access the webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame from webcam")
            break

        movement_message, fall_detected = detector.detect_movement(frame)
        font_color = (0, 0, 255) if fall_detected else (0, 255, 0)
        cv2.putText(frame, movement_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)
        cv2.imshow('Webcam Feed - Fall Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
