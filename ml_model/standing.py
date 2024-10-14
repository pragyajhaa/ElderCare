import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize MediaPipe holistic model
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh  # Import face mesh module

# Initialize Video Capture
cap = cv2.VideoCapture(0)

# Track posture history
posture_history = deque(maxlen=100)  # Store the last 100 postures
abnormal_posture_alerted = False
movement_patterns = deque(maxlen=100)  # Store the last 100 movement patterns

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
            print("Alert: Prolonged abnormal posture detected!")
            abnormal_posture_alerted = True
    else:
        abnormal_posture_alerted = False

    # Analyze movement patterns
    movement_patterns.append(posture)
    if len(set(movement_patterns)) > 2:  # Sudden change in movement pattern
        print("Warning: Sudden change in movement pattern detected!")

# Function to draw key facial landmarks
def draw_key_face_landmarks(image, face_landmarks):
    key_landmark_indices = [1, 33, 263, 61, 291]  # Nose tip, left eye, right eye, left lip, right lip

    if face_landmarks:
        for idx in key_landmark_indices:
            x = int(face_landmarks.landmark[idx].x * image.shape[1])
            y = int(face_landmarks.landmark[idx].y * image.shape[0])
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

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
        
        # Analyze posture
        if results.pose_landmarks:
            posture = analyze_posture(results.pose_landmarks.landmark)
            detect_irregular_movement(posture)
            cv2.putText(image, posture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display the image
        cv2.imshow('Posture and Activity Recognition', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
