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
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Buffer to store detected letters
        self.letter_buffer = deque(maxlen=5)
        self.last_letter_time = time.time()
        
        # Map all letters to their detection functions
        # self.letter_configurations = {
        #     letter: getattr(self, f'detect_{letter.lower()}')
        #     for letter in 'ABCDE'
        #     # for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        # }
        # self.letter_configurations = {
        #     'A': self.detect_a,
        #     'B': self.detect_b,
        #     'C': self.detect_c  # Add this line
        # }
        #self.detect_letter(self.hand.hand_landmarks)

    def detect_letter(self, hand_landmarks):
        # Define rules for detecting letter "E"
        # Convert normalized coordinates to pixel coordinates
        # h, w, _ = image_shape
        thumb_tip = hand_landmarks.landmark[4]    # Thumb tip
        index_tip = hand_landmarks.landmark[8]    # Index finger tip
        middle_tip = hand_landmarks.landmark[12]  # Middle finger tip
        ring_tip = hand_landmarks.landmark[16]    # Ring finger tip
        pinky_tip = hand_landmarks.landmark[20]   # Pinky finger tip

        # Check if fingertips are close to the palm
        finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
        palm_base = hand_landmarks.landmark[0]  # Wrist or palm base

        # Calculate distances and angles
        def distance(lm1, lm2):
            return ((lm1.x - lm2.x) ** 2 + (lm1.y - lm2.y) ** 2) ** 0.5

        def is_curled(tip, base):
            return distance(tip, base) < 0.05  # Adjust threshold as needed

        # Detect letter E: fingers curled, thumb across the front
        close_to_palm = all(is_curled(tip, palm_base) for tip in finger_tips)
        thumb_position = thumb_tip.y > max(tip.y for tip in finger_tips)
        # Detect letter B: fingers straight, palm facing out
        fingers_straight = all(tip.y < palm_base.y for tip in finger_tips)
        thumb_across_palm = thumb_tip.x < palm_base.x
        # Detect letter C: fingers form a curve
        finger_curve = (distance(index_tip, thumb_tip) > 0.1) and (distance(index_tip, middle_tip) > 0.05)
        thumb_position_c = thumb_tip.x < index_tip.x and thumb_tip.y > index_tip.y
        # Detect letter A: thumb tucked, index and middle straight
        thumb_tucked = thumb_tip.x > index_tip.x and thumb_tip.x > middle_tip.x
        thumb_alongside_fingers = thumb_tip.x > palm_base.x and thumb_tip.y < palm_base.y
        # Detect letter D: index finger extended, other fingers curled
        index_extended = not is_curled(index_tip, palm_base)
        other_fingers_curled = all(is_curled(tip, palm_base) for tip in [middle_tip, ring_tip, pinky_tip])
        thumb_touching_middle = distance(thumb_tip, middle_tip) < 0.02  # Adjust threshold as needed

        # Adjust threshold based on empirical observations or hand size
        distance_threshold = 0.08

        close_to_palm = all(
            ((tip.x - palm_base.x) ** 2 + (tip.y - palm_base.y) ** 2) < distance_threshold
            for tip in finger_tips
        )

        # Check if thumb is below the curled fingers
        thumb_below_fingers = thumb_tip.y > max(tip.y for tip in finger_tips)

        is_letter_e = close_to_palm and thumb_across_palm
        is_letter_b = fingers_straight and thumb_across_palm
        is_letter_c = finger_curve and thumb_across_palm
        # not close_to_palm and finger_curve
        is_letter_a = close_to_palm and thumb_alongside_fingers
        is_letter_d = index_extended and other_fingers_curled and thumb_touching_middle

        # Enhanced debugging
        #if confidence > 0.3:  # Lower threshold for debugging
        print("\nDetected")
        print(f"A: {is_letter_a}")
        print(f"B: {is_letter_b}")
        print(f"C: {is_letter_c}")
        print(f"D: {is_letter_d}")
        print(f"E: {is_letter_e}")

        if is_letter_e:
            return "E"
        elif is_letter_b:
            return "B"
        elif is_letter_c:
            return "C"
        elif is_letter_a:
            return "A"
        elif is_letter_d:
            return "D"
        else:
            return None
    # def detect_a(self, hand_landmarks):
    #     # All fingers straight up, thumb tucked
    #     # Fist with thumb to the side
    #     fingers = self.get_finger_states(hand_landmarks)
    #     return fingers[0] and not any(fingers[1:])

    # def detect_b(self, hand_landmarks):
    #     fingers = self.get_finger_states(hand_landmarks)
    #     return not fingers[0] and all(fingers[1:])
    def calculate_confidence(self, conditions, weights):
        """
        Calculate a weighted confidence score based on multiple conditions
        
        Args:
            conditions (list): List of boolean conditions
            weights (list): List of weights for each condition
            
        Returns:
            float: Confidence score between 0 and 1
        """
        if len(conditions) != len(weights):
            raise ValueError("Number of conditions must match number of weights")
            
        if not conditions or not weights:
            return 0.0
            
        # Convert booleans to 1s and 0s and multiply by weights
        weighted_values = [1.0 if cond else 0.0 for cond in conditions]
        weighted_sum = sum(w * v for w, v in zip(weights, weighted_values))
        
        # Normalize by sum of weights to get value between 0 and 1
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
            
        confidence = weighted_sum / total_weight
        
        return confidence

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
                self.draw_debug_info(frame, hand_landmarks) 

                # Check for letters
                current_time = time.time()
                if current_time - self.last_letter_time > 0.5:  # Delay between detections
                    letter = self.detect_letter(hand_landmarks)
                    if letter:
                        print(f"Detected letter: {letter}")
    
        # Display detected letters
        detected_text = ''.join(self.letter_buffer)
        cv2.putText(frame, f"Spelled: {detected_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame

    def draw_debug_info(self, frame, hand_landmarks):
        """
        Draw debug information on the frame
        """
        h, w, _ = frame.shape
        
        # Convert normalized coordinates to pixel coordinates
        def norm_to_pixel(landmark):
            return (
                int(landmark.x * w),  # x coordinate
                int(landmark.y * h)   # y coordinate
            )

        # Draw points and labels
        for idx, landmark in enumerate(hand_landmarks.landmark):
            px, py = norm_to_pixel(landmark)
            cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
            cv2.putText(frame, str(idx), (px+10, py+10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

def main():
    detector = ASLFingerSpellingDetector()

    # Try different camera indices if 0 doesn't work
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
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
