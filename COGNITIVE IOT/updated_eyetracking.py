"""
DIAGNOSTIC V2.0: MediaPipe Face Mesh (EAR + MAR + MOE)
INPUT: Laptop Webcam
LOGIC: Cognitive Load vs. Fatigue Discrimination
"""

import cv2
import mediapipe as mp
import math
import time

# --- Configuration ---
CAMERA_INDEX = 0  
EAR_THRESHOLD = 0.25  
CONSECUTIVE_FRAMES = 2  

# --- MediaPipe Landmark Indices ---
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Inner lip landmarks: [Left Corner, Top1, Top2, Top3, Right Corner, Bot3, Bot2, Bot1]
INNER_LIPS = [78, 82, 13, 312, 308, 317, 14, 87]

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def euclidean_distance(p1, p2):
    return math.dist([p1.x, p1.y], [p2.x, p2.y])

def calculate_ear(landmarks, eye_indices):
    p1, p2, p3 = landmarks[eye_indices[0]], landmarks[eye_indices[1]], landmarks[eye_indices[2]]
    p4, p5, p6 = landmarks[eye_indices[3]], landmarks[eye_indices[4]], landmarks[eye_indices[5]]
    
    v1 = euclidean_distance(p2, p6)
    v2 = euclidean_distance(p3, p5)
    h1 = euclidean_distance(p1, p4)
    
    # EAR formula
    return (v1 + v2) / (2.0 * h1)

def calculate_mar(landmarks, lip_indices):
    p1 = landmarks[lip_indices[0]] # Left corner
    p2 = landmarks[lip_indices[1]] # Top 1
    p3 = landmarks[lip_indices[2]] # Top 2 (Center)
    p4 = landmarks[lip_indices[3]] # Top 3
    p5 = landmarks[lip_indices[4]] # Right corner
    p6 = landmarks[lip_indices[5]] # Bot 3
    p7 = landmarks[lip_indices[6]] # Bot 2 (Center)
    p8 = landmarks[lip_indices[7]] # Bot 1
    
    # Vertical distances
    v1 = euclidean_distance(p2, p8)
    v2 = euclidean_distance(p3, p7)
    v3 = euclidean_distance(p4, p6)
    
    # Horizontal distance
    h1 = euclidean_distance(p1, p5)
    
    # MAR formula
    return (v1 + v2 + v3) / (2.0 * h1)

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    blink_counter = 0
    frame_counter = 0
    
    print("SYSTEM READY: EAR, MAR, and MOE Tracking Initiated.")
    print("Press 'q' in the video window to exit.")

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
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            
            avg_ear = 0.0
            mar = 0.0
            moe = 0.0
            status = "FOCUS / NORMAL"
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 1. Calculate EAR
                    left_ear = calculate_ear(face_landmarks.landmark, LEFT_EYE)
                    right_ear = calculate_ear(face_landmarks.landmark, RIGHT_EYE)
                    avg_ear = (left_ear + right_ear) / 2.0
                    
                    # 2. Calculate MAR
                    mar = calculate_mar(face_landmarks.landmark, INNER_LIPS)
                    
                    # 3. Calculate MOE
                    # Prevent division by zero if EAR drops completely to 0.0
                    if avg_ear > 0.01:
                        moe = mar / avg_ear
                    else:
                        moe = mar / 0.01
                    
                    # Blink Logic
                    if avg_ear < EAR_THRESHOLD:
                        frame_counter += 1
                    else:
                        if frame_counter >= CONSECUTIVE_FRAMES:
                            blink_counter += 1
                        frame_counter = 0

                    # State Logic (Basic thresholding for testing)
                    if moe > 2.5: # Yawn / Fatigue threshold
                        status = "FATIGUE DETECTED (HIGH MOE)"
                    elif avg_ear < EAR_THRESHOLD:
                        status = "BLINKING"
                    
                    # Visual Output (Strict B&W Formatting)
                    cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"MAR: {mar:.2f}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"MOE: {moe:.2f}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"BLINKS: {blink_counter}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"STATE: {status}", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Grayscale conversion constraint
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Draw a black rectangle behind text for high contrast readability
            cv2.rectangle(gray_frame, (15, 15), (420, 190), (0, 0, 0), -1)
            
            # Re-draw text over the black box purely in white
            if results.multi_face_landmarks:
                cv2.putText(gray_frame, f"EAR: {avg_ear:.2f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(gray_frame, f"MAR: {mar:.2f}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(gray_frame, f"MOE: {moe:.2f}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(gray_frame, f"BLINKS: {blink_counter}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(gray_frame, f"STATE: {status}", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Diagnostic V2: Vision Tracking', gray_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()