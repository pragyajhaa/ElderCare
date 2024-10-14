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
        self.fall_frame_count = 0  # Count the number of frames meeting fall criteria
        self.fall_threshold_frames = 5  # Require fall criteria to be met for 5 consecutive frames
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

    def draw_landmarks(self, frame):
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
            
            # Fall detection logic based on angle and speed
            shoulder_left = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            shoulder_right = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            angle = self.calculate_angle(shoulder_left, shoulder_right, hip_left, hip_right)
            
            if vertical_speed > self.velocity_threshold and angle < self.angle_threshold:
                self.fall_frame_count += 1  # Increment fall frame count
                if self.fall_frame_count >= self.fall_threshold_frames:
                    self.fall_frame_count = 0  # Reset after detection
                    return "Fall Detected"
                return "Monitoring Posture..."
            else:
                self.fall_frame_count = 0  # Reset if criteria not met
            
            # Stumble detection based on irregular pattern
            if abs(com_x - (self.prev_com_y if self.prev_com_y else com_x)) > 0.1:
                return "Stumble Detected"
            return "Normal Activity"
        
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
