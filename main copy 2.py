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
        self.letter_configurations = {
            letter: getattr(self, f'detect_{letter.lower()}')
            for letter in 'ABCDEO'
            # for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        }
        # self.letter_configurations = {
        #     'A': self.detect_a,
        #     'B': self.detect_b,
        #     'C': self.detect_c  # Add this line
        # }

    # def get_finger_states(self, hand_landmarks):
    #     """Helper function to get finger states (extended or not)"""
    #     finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    #     finger_mids = [6, 10, 14, 18]  # Mid joints
    #     thumb_tip = hand_landmarks.landmark[4]
    #     thumb_mid = hand_landmarks.landmark[3]
    #     palm_base = hand_landmarks.landmark[0]

    #     fingers_extended = []
    #     # Check thumb
        
    #     # Check other fingers
    #     for tip, mid in zip(finger_tips, finger_mids):
    #         if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mid].y:
    #             fingers_extended.append(True)
    #         else:
    #             fingers_extended.append(False)
                
    #     return fingers_extended
  
    def detect_a(self, hand_landmarks):
        """
        Detect ASL letter 'A'
        Fist with thumb pointing up alongside the index finger
        """
        # Get key points
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        thumb_mcp = hand_landmarks.landmark[2]
        index_tip = hand_landmarks.landmark[8]
        index_pip = hand_landmarks.landmark[6]
        index_mcp = hand_landmarks.landmark[5]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        wrist = hand_landmarks.landmark[0]
        
        # Conditions for 'A'
        thumb_up = thumb_tip.y < thumb_ip.y  # Thumb pointing up
        
        # Thumb should be alongside the fist, not curved around front (like in 'C')
        thumb_alongside = abs(thumb_tip.z - index_mcp.z) < 0.05
        
        # Thumb should be relatively straight
        thumb_straight = abs(thumb_tip.x - thumb_mcp.x) < 0.1
        
        # All fingers should be tightly closed
        fingers_closed = all([
            index_tip.y > index_pip.y,    # Index finger closed
            middle_tip.y > index_pip.y,   # Middle finger closed
            ring_tip.y > index_pip.y,     # Ring finger closed
            pinky_tip.y > index_pip.y     # Pinky closed
        ])
        
        # For 'A', thumb should be more to the side, not curved in front like 'C'
        not_c_shape = abs(thumb_tip.x - index_mcp.x) > 0.02
        
        conditions = [
            thumb_up,
            thumb_alongside,
            thumb_straight,
            fingers_closed,
            not_c_shape
        ]
        
        weights = [1.2, 1.5, 1.0, 1.3, 1.4]
        confidence = self.calculate_confidence(conditions, weights)
        
        return confidence > 0.85
        
    def detect_b(self, hand_landmarks):
        """
        Detect ASL letter 'B'
        All fingers straight up and together, thumb tucked
        """
        # Get key points
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        
        index_pip = hand_landmarks.landmark[6]
        middle_pip = hand_landmarks.landmark[10]
        ring_pip = hand_landmarks.landmark[14]
        pinky_pip = hand_landmarks.landmark[18]
        
        wrist = hand_landmarks.landmark[0]
        
        # All fingers should be extended (pointing up)
        fingers_extended = all([
            index_tip.y < index_pip.y,
            middle_tip.y < middle_pip.y,
            ring_tip.y < ring_pip.y,
            pinky_tip.y < pinky_pip.y
        ])
        
        # Fingers should be close together
        fingers_together = all([
            abs(index_tip.x - middle_tip.x) < 0.05,
            abs(middle_tip.x - ring_tip.x) < 0.05,
            abs(ring_tip.x - pinky_tip.x) < 0.05
        ])
        
        # Thumb should be tucked (lower than other fingers)
        thumb_tucked = thumb_tip.y > index_pip.y
        
        # Fingers should be roughly vertical
        vertical_fingers = all([
            abs(index_tip.x - index_pip.x) < 0.07,
            abs(middle_tip.x - middle_pip.x) < 0.07,
            abs(ring_tip.x - ring_pip.x) < 0.07,
            abs(pinky_tip.x - pinky_pip.x) < 0.07
        ])
        
        return fingers_extended and fingers_together and thumb_tucked and vertical_fingers

    def detect_c(self, hand_landmarks):
        """
        Detect ASL letter 'C'
        Alternative method focusing on the C-shape curve and spacing
        """
        # Get all relevant landmarks
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        thumb_mcp = hand_landmarks.landmark[2]
        
        index_tip = hand_landmarks.landmark[8]
        index_pip = hand_landmarks.landmark[6]
        index_mcp = hand_landmarks.landmark[5]
        
        middle_tip = hand_landmarks.landmark[12]
        middle_pip = hand_landmarks.landmark[10]
        
        ring_tip = hand_landmarks.landmark[16]
        ring_pip = hand_landmarks.landmark[14]
        
        pinky_tip = hand_landmarks.landmark[20]
        pinky_pip = hand_landmarks.landmark[18]
        
        # 1. Check if thumb and index form roughly a C shape
        thumb_to_index_dist = abs(thumb_tip.x - index_tip.x)
        proper_c_gap = 0.1 < thumb_to_index_dist < 0.3  # Adjust these values as needed
        
        # 2. Check if fingers are curved but not closed
        fingers_curved = all([
            # Tips should be lower than MCPs but higher than PIPs
            index_mcp.y < index_tip.y < index_pip.y,
            middle_pip.y > middle_tip.y > middle_pip.y * 0.8,
            ring_pip.y > ring_tip.y > ring_pip.y * 0.8,
            pinky_pip.y > pinky_tip.y > pinky_pip.y * 0.8
        ])
        
        # 3. Check if thumb is more horizontal than vertical
        thumb_angle = abs(thumb_tip.y - thumb_ip.y) / abs(thumb_tip.x - thumb_ip.x)
        thumb_horizontal = thumb_angle < 1.0  # Closer to horizontal
        
        # 4. Check if fingers are relatively aligned (forming curve)
        finger_alignment = all([
            abs(index_tip.y - middle_tip.y) < 0.1,
            abs(middle_tip.y - ring_tip.y) < 0.1,
            abs(ring_tip.y - pinky_tip.y) < 0.1
        ])
        
        # 5. Check if thumb is in front
        thumb_forward = thumb_tip.z < index_mcp.z
        
        # 6. Check finger spacing (not too spread, not too tight)
        finger_spacing = all([
            0.02 < abs(index_tip.x - middle_tip.x) < 0.08,
            0.02 < abs(middle_tip.x - ring_tip.x) < 0.08,
            0.02 < abs(ring_tip.x - pinky_tip.x) < 0.08
        ])
        
        conditions = [
            proper_c_gap,      # Weight: 1.5 (crucial for C shape)
            fingers_curved,    # Weight: 1.3 (important for overall shape)
            thumb_horizontal,  # Weight: 1.2 (characteristic of C)
            finger_alignment,  # Weight: 1.0 (helps with form)
            thumb_forward,     # Weight: 1.4 (distinguishes from other letters)
            finger_spacing    # Weight: 1.1 (helps with form)
        ]
        
        weights = [1.5, 1.3, 1.2, 1.0, 1.4, 1.1]
        confidence = self.calculate_confidence(conditions, weights)
        
        # For debugging
        if confidence > 0.5:  # Lower threshold for debugging
            print(f"C Detection Confidence: {confidence:.2f}")
            print(f"Gap: {proper_c_gap}")
            print(f"Curved: {fingers_curved}")
            print(f"Thumb Horizontal: {thumb_horizontal}")
            print(f"Alignment: {finger_alignment}")
            print(f"Thumb Forward: {thumb_forward}")
            print(f"Spacing: {finger_spacing}")
            print("---")
        
        return confidence > 0.80  # Adjust threshold as needed

    def detect_d(self, hand_landmarks):
        """
        Detect ASL letter 'D'
        Index finger straight up, other fingers closed, thumb touches middle finger
        """
        # Get key points
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        thumb_mcp = hand_landmarks.landmark[2]
        
        index_tip = hand_landmarks.landmark[8]
        index_pip = hand_landmarks.landmark[6]
        index_mcp = hand_landmarks.landmark[5]
        
        middle_tip = hand_landmarks.landmark[12]
        middle_pip = hand_landmarks.landmark[10]
        
        ring_tip = hand_landmarks.landmark[16]
        ring_pip = hand_landmarks.landmark[14]
        
        pinky_tip = hand_landmarks.landmark[20]
        pinky_pip = hand_landmarks.landmark[18]

        # 1. Index finger should be straight and pointing up
        index_straight = (
            index_tip.y < index_pip.y and  # Tip above PIP
            abs(index_tip.x - index_pip.x) < 0.04  # Vertically aligned
        )

        # 2. Other fingers should be closed
        other_fingers_closed = all([
            middle_tip.y > middle_pip.y,  # Middle finger closed
            ring_tip.y > ring_pip.y,      # Ring finger closed
            pinky_tip.y > pinky_pip.y     # Pinky closed
        ])

        # 3. Thumb should be touching or very close to middle finger
        thumb_middle_touch = (
            abs(thumb_tip.x - middle_tip.x) < 0.05 and
            abs(thumb_tip.y - middle_tip.y) < 0.05 and
            abs(thumb_tip.z - middle_tip.z) < 0.05
        )

        # 4. Index finger should be separated from other fingers
        index_separated = all([
            abs(index_tip.x - middle_tip.x) > 0.04,
            abs(index_tip.x - ring_tip.x) > 0.04,
            abs(index_tip.x - pinky_tip.x) > 0.04
        ])

        # 5. Index should be vertical
        index_vertical = abs(index_tip.x - index_mcp.x) < 0.08

        conditions = [
            index_straight,     # Weight: 1.5 (crucial for D shape)
            other_fingers_closed,  # Weight: 1.3 (important)
            thumb_middle_touch,    # Weight: 1.4 (characteristic of D)
            index_separated,       # Weight: 1.2 (helps distinguish from U)
            index_vertical        # Weight: 1.1 (helps with form)
        ]
        
        weights = [1.5, 1.3, 1.4, 1.2, 1.1]
        confidence = self.calculate_confidence(conditions, weights)

        # For debugging
        if confidence > 0.5:  # Lower threshold for debugging
            print(f"D Detection Confidence: {confidence:.2f}")
            print(f"Index straight: {index_straight}")
            print(f"Others closed: {other_fingers_closed}")
            print(f"Thumb touch: {thumb_middle_touch}")
            print(f"Index separated: {index_separated}")
            print(f"Index vertical: {index_vertical}")
            print("---")

        return confidence > 0.80

    def detect_e(self, hand_landmarks):
        # Get key points
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        thumb_mcp = hand_landmarks.landmark[2]
        
        index_tip = hand_landmarks.landmark[8]
        index_pip = hand_landmarks.landmark[6]
        index_mcp = hand_landmarks.landmark[5]
        
        middle_tip = hand_landmarks.landmark[12]
        middle_pip = hand_landmarks.landmark[10]
        
        ring_tip = hand_landmarks.landmark[16]
        ring_pip = hand_landmarks.landmark[14]
        
        pinky_tip = hand_landmarks.landmark[20]
        pinky_pip = hand_landmarks.landmark[18]
        
        # 1. All fingers should be curled
        fingers_curled = all([
            index_tip.y > index_pip.y,
            middle_tip.y > middle_pip.y,
            ring_tip.y > ring_pip.y,
            pinky_tip.y > pinky_pip.y
        ])
        
        # 2. Thumb should be across fingers
        # thumb_across = (
        #     thumb_tip.y > middle_tip.y and
        #     thumb_tip.x > middle_pip.x
        # )
        
        # 3. Thumb should be in front of other fingers
        # thumb_in_front = thumb_tip.z < index_pip.z
        
        # 4. Fingers should be close together
        # fingers_close_together = all([
        #     abs(index_tip.x - middle_tip.x) < 0.08,
        #     abs(middle_tip.x - ring_tip.x) < 0.08,
        #     abs(ring_tip.x - pinky_tip.x) < 0.08
        # ])
        fingers_close_together = self.check_fingers_close_together(middle_tip, ring_tip, pinky_tip)

        # 5. Thumb position relative to palm
        thumb_position = (
            thumb_tip.x < index_mcp.x and
            thumb_tip.y > thumb_mcp.y
        )
        
        conditions = [
            fingers_curled,    # Weight: 1.5 (crucial)
            # thumb_across,      # Weight: 1.4 (important)
            #thumb_in_front,    # Weight: 1.2 (helpful)
            fingers_close_together,  # Weight: 1.1 (form)
            thumb_position     # Weight: 1.3 (characteristic)
        ]
        
        weights = [1.5, 1.1, 1.3]
        confidence = self.calculate_confidence(conditions, weights)
        
        # Debug information
        if confidence > 0.3:
            print("\nE Detection Details:")
            print(f"Overall Confidence: {confidence:.2f}")
            print(f"Fingers curled: {fingers_curled}")
            #print(f"Thumb across: {thumb_across}")
            print(f"Fingers together: {fingers_close_together}")
            print(f"Thumb position: {thumb_position}")
            print("---")
        
        return confidence > 0.75  # Adjust threshold as needed

    def detect_o(self, hand_landmarks):
        # Define landmarks within the function
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        index_pip = hand_landmarks.landmark[6]
        middle_tip = hand_landmarks.landmark[12]
        middle_pip = hand_landmarks.landmark[10]
        ring_tip = hand_landmarks.landmark[16]
        ring_pip = hand_landmarks.landmark[14]
        pinky_tip = hand_landmarks.landmark[20]
        pinky_pip = hand_landmarks.landmark[18]
        wrist = hand_landmarks.landmark[0]

        # Check if thumb and index finger are forming a circle
        thumb_index_circle = self.check_thumb_index_circle(thumb_tip, index_tip)

        # Check if other fingers are extended and close together
        other_fingers_extended = self.check_other_fingers_extended(middle_tip, middle_pip, ring_tip, ring_pip, pinky_tip, pinky_pip)
        fingers_close_together = self.check_fingers_close_together(middle_tip, ring_tip, pinky_tip)

        # Check if palm is facing the camera
        palm_facing_camera = self.is_palm_facing_camera(wrist, hand_landmarks.landmark[9])  # Using middle_mcp (landmark 9) for palm orientation

        # Calculate confidence based on these conditions
        conditions = [thumb_index_circle, other_fingers_extended, fingers_close_together, palm_facing_camera]
        weights = [1.5, 1.2, 1.0, 1.3]  # Adjust weights based on importance
        confidence = self.calculate_confidence(conditions, weights)

        # Debug information
        if confidence > 0.3:
            print("\nO Detection Details:")
            print(f"Overall Confidence: {confidence:.2f}")
            print(f"Fingers curled: ")
            #print(f"Thumb across: {thumb_across}")
            print(f"thumb_index_circle across: {thumb_index_circle}")
            print(f"Other fingers extended: {other_fingers_extended}")
            print(f"Fingers together: {fingers_close_together}")
            print(f"Thumb position: ")
            print("---")

        return confidence > 0.75  # Adjust threshold as needed

    def check_thumb_index_circle(self, thumb_tip, index_tip):
        # Calculate distance between thumb tip and index tip
        distance = ((thumb_tip.x - index_tip.x)**2 + 
                    (thumb_tip.y - index_tip.y)**2 + 
                    (thumb_tip.z - index_tip.z)**2)**0.5
        
        # Check if the distance is within a certain range
        return 0.02 < distance < 0.1  # Adjust these thresholds as needed

    def check_other_fingers_extended(self, middle_tip, middle_pip, ring_tip, ring_pip, pinky_tip, pinky_pip):
        return all([
            middle_tip.y < middle_pip.y,
            ring_tip.y < ring_pip.y,
            pinky_tip.y < pinky_pip.y
        ])

    def check_fingers_close_together(self, middle_tip, ring_tip, pinky_tip):
        return all([
            0.02 < abs(middle_tip.x - ring_tip.x) < 0.08,
            0.02 < abs(ring_tip.x - pinky_tip.x) < 0.08
        ])

    def is_palm_facing_camera(self, wrist, middle_mcp):
        return middle_mcp.z < wrist.z

    # def calculate_confidence(self, conditions, weights):
    #     total_weight = sum(weights)
    #     return weighted_sum / total_weight
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
