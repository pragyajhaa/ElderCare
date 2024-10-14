import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
from mediapipe.python.solutions import holistic as mp_holistic

# Initialize MediaPipe holistic model
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh  # Import face mesh module

# Initialize Video Capture
cap = cv2.VideoCapture(0)

# Track posture history and movement patterns
posture_history = deque(maxlen=100)  # Store the last 100 postures
movement_patterns = deque(maxlen=100)  # Store the last 100 movement patterns
abnormal_posture_alerted = False

# Define class for posture tracking and alerts
class PostureTracker:
    def __init__(self):
        self.current_posture = None
        self.posture_start_time = None
        self.posture_durations = {'Sitting': 0, 'Standing': 0, 'Abnormal Posture': 0}

    def update_posture(self, posture):
        current_time = time.time()
        if self.current_posture:
            duration = current_time - self.posture_start_time
            self.posture_durations[self.current_posture] += duration

        if posture != self.current_posture:
            self.current_posture = posture
            self.posture_start_time = current_time

    def get_posture_durations(self):
        return self.posture_durations

# Define class for movement detection
class MovementDetector:
    def __init__(self, movement_threshold=0.1):
        self.movement_threshold = movement_threshold
        self.previous_landmarks = []

    def detect_movement(self, landmarks):
        if not self.previous_landmarks:
            self.previous_landmarks = landmarks
            return "No movement detected"

        current_landmarks = [np.array([landmark.x, landmark.y, landmark.z]) for landmark in landmarks]
        previous_landmarks = [np.array([landmark.x, landmark.y, landmark.z]) for landmark in self.previous_landmarks]

        distances = [np.linalg.norm(current - prev) for current, prev in zip(current_landmarks, previous_landmarks)]
        self.previous_landmarks = landmarks

        if max(distances) > self.movement_threshold:
            return "Movement detected"
        else:
            return "No movement detected"
# Define class for seizure detection
# class SeizureDetector:
#     def __init__(self):
#         self.prev_angles = {'left_shoulder': None, 'right_shoulder': None}
#         self.seizure_threshold = 0.5  # Adjust as needed

#     def detect_seizures(self, angles):
#         seizure_detected = False

#         for key, angle in angles.items():
#             if self.prev_angles[key] is not None:
#                 if abs(angle - self.prev_angles[key]) > self.seizure_threshold:
#                     seizure_detected = True
#                     break

#             self.prev_angles[key] = angle

#         return "Seizure Detected" if seizure_detected else "No Seizure Detected"

# Define class for immobility detection
class ImmobilityDetector:
    def __init__(self, immobility_threshold=60):
        self.last_movement_time = time.time()
        self.immobility_threshold = immobility_threshold

    def update_movement_time(self):
        self.last_movement_time = time.time()

    def detect_immobility(self):
        current_time = time.time()
        if current_time - self.last_movement_time > self.immobility_threshold:
            return "Immobility Detected"
        return "No Immobility Detected"

# Define class for notifications
class Notifier:
    def send_notification(self, message):
        # Replace with actual notification logic (e.g., email, SMS, app alert)
        print(f"Notification: {message}")

# Define functions to calculate angles between joints
def calculate_angle(a, b, c):
    a = np.array(a)  # First joint
    b = np.array(b)  # Middle joint
    c = np.array(c)  # End joint
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle

# Analyze posture based on key angles
def analyze_posture(landmarks):
    left_shoulder = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]
    
    right_shoulder = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_elbow = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist = [landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y]
    
    left_shoulder_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_shoulder_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # Define angle thresholds for sitting and standing
    sitting_threshold_low = 50
    sitting_threshold_high = 130
    standing_threshold_low = 160
    standing_threshold_high = 180

    # Analyze posture based on angles
    if sitting_threshold_low < left_shoulder_angle < sitting_threshold_high and \
       sitting_threshold_low < right_shoulder_angle < sitting_threshold_high:
        return "Sitting"
    elif standing_threshold_low < left_shoulder_angle < standing_threshold_high and \
         standing_threshold_low < right_shoulder_angle < standing_threshold_high:
        return "Standing"
    else:
        return "Abnormal Posture"

# Detect irregular movement patterns
def detect_irregular_movement(posture):
    global abnormal_posture_alerted
    
    posture_history.append(posture)
    
    # Check for prolonged abnormal posture
    if posture_history.count("Abnormal Posture") > len(posture_history) * 0.6:  # 60% abnormal
        if not abnormal_posture_alerted:
            notifier.send_notification("Prolonged abnormal posture detected!")
            abnormal_posture_alerted = True
    else:
        abnormal_posture_alerted = False

    # Analyze movement patterns
    movement_patterns.append(posture)
    if len(set(movement_patterns)) > 2:  # Sudden change in movement pattern
        notifier.send_notification("Sudden change in movement pattern detected!")

# Function to draw key facial landmarks
def draw_key_face_landmarks(image, face_landmarks):
    key_landmark_indices = [1, 33, 263, 61, 291]  # Nose tip, left eye, right eye, left lip, right lip

    if face_landmarks:
        for idx in key_landmark_indices:
            x = int(face_landmarks.landmark[idx].x * image.shape[1])
            y = int(face_landmarks.landmark[idx].y * image.shape[0])
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

# Initialize trackers
posture_tracker = PostureTracker()
movement_detector = MovementDetector()
#seizure_detector = SeizureDetector()
immobility_detector = ImmobilityDetector()
notifier = Notifier()

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Recolor the frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detections
        results = holistic.process(image)
        
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw reduced face landmarks
        if results.face_landmarks:
            draw_key_face_landmarks(image, results.face_landmarks)
        
        # Draw body landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        # Analyze posture and detect movements
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            posture = analyze_posture(landmarks)
            posture_tracker.update_posture(posture)
            
            # Detect movements
            key_landmarks = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value],
                             landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]]
            movement_status = movement_detector.detect_movement(key_landmarks)
            cv2.putText(image, movement_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Detect seizures
            angles = {'left_shoulder': calculate_angle([landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,
                                                        landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,
                                                        landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x],
                                                        [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,
                                                         landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x],
                                                        [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x, 0]),
                      'right_shoulder': calculate_angle([landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x,
                                                         landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x,
                                                         landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x],
                                                         [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x,
                                                          landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x],
                                                         [landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x, 0])}
            #seizure_status = seizure_detector.detect_seizures(angles)
            #cv2.putText(image, seizure_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Detect immobility
            immobility_status = immobility_detector.detect_immobility()
            cv2.putText(image, immobility_status, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Update immobility detector on each frame
            immobility_detector.update_movement_time()
        
        # Detect irregular movements
        detect_irregular_movement(posture)

        # Display the image
        cv2.imshow('Posture and Activity Recognition', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
