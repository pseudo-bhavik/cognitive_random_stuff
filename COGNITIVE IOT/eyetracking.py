"""
DIAGNOSTIC: MediaPipe Face Mesh Blink Detection
INPUT: DroidCam Virtual Camera (USB Hardwired)
"""

import cv2
import mediapipe as mp
import math
import time

# --- Configuration ---
CAMERA_INDEX = 0  # Adjust if DroidCam is mapped to a different index (0, 1, or 2)
EAR_THRESHOLD = 0.25  # Tune this based on lighting and user eye shape
CONSECUTIVE_FRAMES = 2  # Frames the eye must be closed to register a blink

# MediaPipe Face Mesh landmark indices for eyes
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def euclidean_distance(p1, p2):
    return math.dist([p1.x, p1.y], [p2.x, p2.y])

def calculate_ear(landmarks, eye_indices):
    # p1, p2, p3, p4, p5, p6
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]
    
    # Calculate vertical distances
    v1 = euclidean_distance(p2, p6)
    v2 = euclidean_distance(p3, p5)
    # Calculate horizontal distance
    h1 = euclidean_distance(p1, p4)
    
    # EAR formula
    ear = (v1 + v2) / (2.0 * h1)
    return ear

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open video source. Check CAMERA_INDEX and DroidCam connection.")
        return

    blink_counter = 0
    frame_counter = 0
    
    print("SYSTEM READY: Vision tracking initiated. Press 'q' to exit.")

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Calculate EAR for both eyes
                    left_ear = calculate_ear(face_landmarks.landmark, LEFT_EYE)
                    right_ear = calculate_ear(face_landmarks.landmark, RIGHT_EYE)
                    avg_ear = (left_ear + right_ear) / 2.0
                    
                    # Blink Logic
                    if avg_ear < EAR_THRESHOLD:
                        frame_counter += 1
                    else:
                        if frame_counter >= CONSECUTIVE_FRAMES:
                            blink_counter += 1
                            print(f"BLINK REGISTERED | Total: {blink_counter} | Timestamp: {time.strftime('%H:%M:%S')}")
                        frame_counter = 0
                    
                    # Visual Output (Strictly B&W overlay style for diagnostics)
                    cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"BLINKS: {blink_counter}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Grayscale conversion for diagnostic aesthetic constraint
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Diagnostic: Blink Tracking', gray_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()




