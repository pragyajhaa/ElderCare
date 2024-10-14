import cv2
import mediapipe as mp
import numpy as np
import time

class MovementDetection:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.prev_com_y = None  # Previous center of mass y-coordinate
        self.prev_time = None  # Time when the previous frame was processed
        self.fall_start_time = None
        self.immobility_start_time = None
        self.immobility_threshold = 5  # seconds
        self.vertical_threshold = 100  # Pixels
        self.velocity_threshold = 50  # Pixels/sec
        self.angle_threshold = 45  # Degrees
        self.repetitive_motion = []  # Store recent motions for seizure detection
        
    def detect_movement(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            self.draw_landmarks(frame, landmarks)
            movement_type = self.analyze_movement(landmarks)
            return movement_type
        else:
            return "Normal Activity"
        
    def draw_landmarks(self, frame, landmarks):
        for landmark in landmarks:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    def analyze_movement(self, landmarks):
        # Calculate center of mass (COM) as average of hips
        hip_left = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        hip_right = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        com_x = (hip_left.x + hip_right.x) / 2
        com_y = (hip_left.y + hip_right.y) / 2
        
        # Calculate time delta
        current_time = time.time()
        if self.prev_time is not None:
            time_delta = current_time - self.prev_time
            # Calculate vertical speed
            vertical_speed = abs(com_y - self.prev_com_y) / time_delta
            
            # Store movement for seizure detection
            self.repetitive_motion.append(vertical_speed)
            if len(self.repetitive_motion) > 20:
                self.repetitive_motion.pop(0)  # Keep last 20 movements
            
            seizure_score = np.mean(self.repetitive_motion)
            
            # Seizure detection logic
            if seizure_score > self.velocity_threshold:
                return "Seizure Detected"
            
            # Fall detection logic
            if vertical_speed > self.velocity_threshold:
                shoulder_left = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                shoulder_right = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                angle = self.calculate_angle(shoulder_left, shoulder_right, hip_left, hip_right)
                
                if angle < self.angle_threshold:
                    if self.fall_start_time is None:
                        self.fall_start_time = current_time
                    elif current_time - self.fall_start_time > 1:  # Stable detection
                        self.fall_start_time = None
                        return "Fall Detected"
                    return "Monitoring Posture..."
                else:
                    # Stumble detection based on irregular pattern
                    if abs(com_x - (self.prev_com_y if self.prev_com_y else com_x)) > 0.1:
                        return "Stumble Detected"
                    return "Normal Activity"
            else:
                self.fall_start_time = None
        
        # Update previous COM and time
        self.prev_com_y = com_y
        self.prev_time = current_time
        return "Normal Activity"

    def calculate_angle(self, shoulder_left, shoulder_right, hip_left, hip_right):
        # Calculate angle of the line connecting shoulders and hips with respect to horizontal
        shoulder_mid_x = (shoulder_left.x + shoulder_right.x) / 2
        shoulder_mid_y = (shoulder_left.y + shoulder_right.y) / 2
        hip_mid_x = (hip_left.x + hip_right.x) / 2
        hip_mid_y = (hip_left.y + hip_right.y) / 2
        
        angle = np.degrees(np.arctan2(hip_mid_y - shoulder_mid_y, hip_mid_x - shoulder_mid_x))
        return abs(angle)
    
def main():
    cap = cv2.VideoCapture(0)
    detector = MovementDetection()

    if not cap.isOpened():
        print("Error: Cannot access the webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read the frame")
            break
        
        movement_type = detector.detect_movement(frame)
        
        cv2.putText(frame, movement_type, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (0, 0, 255) if "Detected" in movement_type else (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Movement Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
