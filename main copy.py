import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

class ASLFingerSpellingDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Buffer to store detected letters
        self.letter_buffer = deque(maxlen=5)
        self.last_letter_time = time.time()
        
        # Simple mapping of hand configurations to letters
        # This is a basic example - you would need more complex logic for accurate detection
        self.letter_configurations = {
            'A': self.detect_a,
            'B': self.detect_b,
            # Add more letters with their corresponding detection functions
        }
        
    def detect_a(self, hand_landmarks):
        # Example detection for letter 'A'
        # Thumb up, other fingers closed
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        
        if thumb_tip.y < index_tip.y:
            return True
        return False
    
    def detect_b(self, hand_landmarks):
        # Example detection for letter 'B'
        # All fingers straight up
        finger_tips = [hand_landmarks.landmark[i] for i in [8, 12, 16, 20]]
        palm_base = hand_landmarks.landmark[0]
        
        all_fingers_up = all(tip.y < palm_base.y for tip in finger_tips)
        return all_fingers_up
    
    def process_frame(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Check for letters
                current_time = time.time()
                if current_time - self.last_letter_time > 0.5:  # Delay between detections
                    for letter, detect_func in self.letter_configurations.items():
                        if detect_func(hand_landmarks):
                            self.letter_buffer.append(letter)
                            self.last_letter_time = current_time
                            break
        
        # Display detected letters
        detected_text = ''.join(self.letter_buffer)
        cv2.putText(frame, f"Spelled: {detected_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame

def main():
    detector = ASLFingerSpellingDetector()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process the frame
        processed_frame = detector.process_frame(frame)
        
        # Display the frame
        cv2.imshow('ASL Fingerspelling Detection', processed_frame)
        
        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
