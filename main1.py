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
        self.letter_buffer = deque(maxlen=5)
        self.last_letter_time = time.time()
        
        # Map all letters to their detection functions
        self.letter_configurations = {
            letter: getattr(self, f'detect_{letter.lower()}')
            for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        }

    def get_hand_metrics(self, hand_landmarks):
        """Calculate basic hand metrics for normalization"""
        wrist = hand_landmarks.landmark[0]
        hand_size = np.sqrt(
            (hand_landmarks.landmark[17].x - hand_landmarks.landmark[1].x)**2 +
            (hand_landmarks.landmark[17].y - hand_landmarks.landmark[1].y)**2
        )
        return wrist, hand_size

    def normalized_distance(self, p1, p2, hand_size):
        """Calculate normalized distance between two points"""
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2) / hand_size

    def get_finger_curve_angle(self, tip, mid, base):
        """Calculate angle between three points"""
        vector1 = np.array([tip.x - mid.x, tip.y - mid.y])
        vector2 = np.array([base.x - mid.x, base.y - mid.y])
        
        # Handle zero vectors
        if np.all(vector1 == 0) or np.all(vector2 == 0):
            return 0
            
        vector1 = vector1 / np.linalg.norm(vector1)
        vector2 = vector2 / np.linalg.norm(vector2)
        
        dot_product = np.clip(np.dot(vector1, vector2), -1.0, 1.0)
        angle = np.arccos(dot_product)
        return np.degrees(angle)

    def get_finger_states(self, hand_landmarks):
        """
        Get detailed finger state information
        Returns: Dictionary with finger states and angles
        """
        # Landmark indices
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
        finger_pips = [6, 10, 14, 18]  # Second joints
        finger_mcps = [5, 9, 13, 17]   # Base joints
        
        
        finger_info = {
            'thumb': {
                'extended': False,
                'angle': 0,
                'tip_pos': hand_landmarks.landmark[4]
            }
        }
        
        # Analyze thumb
        thumb_angle = self.get_finger_curve_angle(
            hand_landmarks.landmark[4],  # tip
            hand_landmarks.landmark[3],  # ip
            hand_landmarks.landmark[2]   # mcp
        )
        finger_info['thumb'].update({
            'extended': thumb_angle > 30,
            'angle': thumb_angle
        })
        
        # Analyze other fingers
        finger_names = ['index', 'middle', 'ring', 'pinky']
        for name, tip, pip, mcp in zip(finger_names, finger_tips, finger_pips, finger_mcps):
            tip_landmark = hand_landmarks.landmark[tip]
            pip_landmark = hand_landmarks.landmark[pip]
            mcp_landmark = hand_landmarks.landmark[mcp]
            
            angle = self.get_finger_curve_angle(tip_landmark, pip_landmark, mcp_landmark)
            vertical_extension = tip_landmark.y < pip_landmark.y
            
            finger_info[name] = {
                'extended': vertical_extension and angle > 20,
                'angle': angle,
                'tip_pos': tip_landmark,
                'pip_pos': pip_landmark,
                'mcp_pos': mcp_landmark
            }
        
        return finger_info

    def calculate_confidence(self, conditions, weights=None):
        """Calculate confidence score based on multiple conditions"""
        if weights is None:
            weights = [1.0] * len(conditions)
        
        total_weight = sum(weights)
        confidence = sum(c * w for c, w in zip(conditions, weights)) / total_weight
        return confidence
    
    def process_frame(self, frame):
        """
        Process a single frame and detect ASL letters
        """
        # Convert the frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Hands
        results = self.hands.process(rgb_frame)
        
        # Draw hand landmarks and process detections
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the landmarks
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Check for each letter
                confidences = {
                    'T': self.detect_t(hand_landmarks),
                    'U': self.detect_u(hand_landmarks),
                    'V': self.detect_v(hand_landmarks),
                    'W': self.detect_w(hand_landmarks),
                    'X': self.detect_x(hand_landmarks),
                    'Y': self.detect_y(hand_landmarks),
                    'Z': self.detect_z(hand_landmarks)
                }
                
                # Find the letter with highest confidence
                detected_letter = max(confidences.items(), key=lambda x: x[1])
                if detected_letter[1] > 0.85:  # Confidence threshold
                    # Draw the detected letter on the frame
                    cv2.putText(
                        frame,
                        f"Letter: {detected_letter[0]}",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
        
        return frame
    
    def detect_a(self, hand_landmarks):
        """
        Detect ASL letter 'A'
        Fist with thumb extended to the side
        """
        finger_info = self.get_finger_states(hand_landmarks)
        wrist, hand_size = self.get_hand_metrics(hand_landmarks)
        
        # Get thumb position
        thumb_tip = finger_info['thumb']['tip_pos']
        index_tip = finger_info['index']['tip_pos']
        
        conditions = [
            not finger_info['index']['extended'],    # Index closed
            not finger_info['middle']['extended'],   # Middle closed
            not finger_info['ring']['extended'],     # Ring closed
            not finger_info['pinky']['extended'],    # Pinky closed
            finger_info['thumb']['extended'],        # Thumb extended
            thumb_tip.x < index_tip.x,              # Thumb to side
            finger_info['thumb']['angle'] > 30       # Thumb angle appropriate
        ]
        
        weights = [1.0, 1.0, 1.0, 1.0, 1.5, 1.2, 1.2]
        confidence = self.calculate_confidence(conditions, weights)
        
        return confidence > 0.8

    def detect_b(self, hand_landmarks):
        """
        Detect ASL letter 'B'
        All fingers straight up, thumb tucked
        """
        finger_info = self.get_finger_states(hand_landmarks)
        
        # Check vertical alignment of fingers
        finger_tips = [finger_info[name]['tip_pos'] for name in ['index', 'middle', 'ring', 'pinky']]
        x_positions = [tip.x for tip in finger_tips]
        x_variance = np.var(x_positions)
        
        conditions = [
            finger_info['index']['extended'],     # Index extended
            finger_info['middle']['extended'],    # Middle extended
            finger_info['ring']['extended'],      # Ring extended
            finger_info['pinky']['extended'],     # Pinky extended
            not finger_info['thumb']['extended'], # Thumb tucked
            x_variance < 0.01                     # Fingers aligned vertically
        ]
        
        weights = [1.0, 1.0, 1.0, 1.0, 1.2, 1.5]
        confidence = self.calculate_confidence(conditions, weights)
        
        return confidence > 0.8

    def detect_c(self, hand_landmarks):
        """
        Detect ASL letter 'C'
        Curved hand shape
        """
        finger_info = self.get_finger_states(hand_landmarks)
        
        # Get key positions
        thumb_tip = finger_info['thumb']['tip_pos']
        index_tip = finger_info['index']['tip_pos']
        pinky_tip = finger_info['pinky']['tip_pos']
        
        # Calculate curve metrics
        thumb_to_index = self.normalized_distance(thumb_tip, index_tip, hand_size)
        index_to_pinky = self.normalized_distance(index_tip, pinky_tip, hand_size)
        
        # Check curved position
        conditions = [
            0.1 < thumb_to_index < 0.3,          # Appropriate thumb-index spacing
            0.2 < index_to_pinky < 0.4,          # Appropriate finger spread
            thumb_tip.x < index_tip.x,           # Thumb position correct
            30 < finger_info['thumb']['angle'] < 90,  # Thumb curved appropriately
            all(20 < finger_info[name]['angle'] < 60  # Fingers curved but not closed
                for name in ['index', 'middle', 'ring', 'pinky'])
        ]
        
        weights = [1.5, 1.2, 1.0, 1.3, 1.5]
        confidence = self.calculate_confidence(conditions, weights)
        
        return confidence > 0.75

    def detect_d(self, hand_landmarks):
        """
        Detect ASL letter 'D'
        Index finger pointing up, others curved
        """
        finger_info = self.get_finger_states(hand_landmarks)
        
        index_straight = finger_info['index']['angle'] > 150  # Index should be straight
        
        conditions = [
            finger_info['index']['extended'],     # Index extended
            index_straight,                       # Index straight
            not finger_info['middle']['extended'],# Middle curved
            not finger_info['ring']['extended'],  # Ring curved
            not finger_info['pinky']['extended'], # Pinky curved
            not finger_info['thumb']['extended']  # Thumb tucked
        ]
        
        weights = [1.5, 1.2, 1.0, 1.0, 1.0, 1.2]
        confidence = self.calculate_confidence(conditions, weights)
        
        return confidence > 0.8

    def detect_e(self, hand_landmarks):
        """
        Detect ASL letter 'E'
        All fingers curled, thumb tucked
        """
        finger_info = self.get_finger_states(hand_landmarks)
        
        conditions = [
            not finger_info['thumb']['extended'],  # Thumb tucked
            not finger_info['index']['extended'],  # Index curled
            not finger_info['middle']['extended'], # Middle curled
            not finger_info['ring']['extended'],   # Ring curled
            not finger_info['pinky']['extended'],  # Pinky curled
            all(finger_info[name]['angle'] < 45    # All fingers tightly curled
                for name in ['index', 'middle', 'ring', 'pinky'])
        ]
        
        weights = [1.2, 1.0, 1.0, 1.0, 1.0, 1.5]
        confidence = self.calculate_confidence(conditions, weights)
        
        return confidence > 0.85

    def detect_f(self, hand_landmarks):
        """
        Detect ASL letter 'F'
        Index and thumb touching, other fingers extended
        """
        finger_info = self.get_finger_states(hand_landmarks)
        
        # Check thumb and index touching
        thumb_tip = finger_info['thumb']['tip_pos']
        index_tip = finger_info['index']['tip_pos']
        thumb_index_dist = self.normalized_distance(thumb_tip, index_tip, hand_size)
        
        conditions = [
            thumb_index_dist < 0.1,               # Thumb and index touching
            finger_info['middle']['extended'],    # Middle extended
            finger_info['ring']['extended'],      # Ring extended
            finger_info['pinky']['extended'],     # Pinky extended
            finger_info['thumb']['angle'] > 30    # Thumb positioned correctly
        ]
        
        weights = [1.5, 1.0, 1.0, 1.0, 1.2]
        confidence = self.calculate_confidence(conditions, weights)
        
        return confidence > 0.8

    def detect_g(self, hand_landmarks):
        """
        Detect ASL letter 'G'
        Index finger pointing to the side, thumb extended
        """
        finger_info = self.get_finger_states(hand_landmarks)
        
        # Check index finger horizontal position
        index_tip = finger_info['index']['tip_pos']
        index_mcp = finger_info['index']['mcp_pos']
        index_horizontal = abs(index_tip.x - index_mcp.x) > 0.1
        
        conditions = [
            finger_info['index']['extended'],     # Index extended
            index_horizontal,                     # Index pointing sideways
            not finger_info['middle']['extended'],# Middle curled
            not finger_info['ring']['extended'],  # Ring curled
            not finger_info['pinky']['extended'], # Pinky curled
            finger_info['thumb']['extended']      # Thumb extended
        ]
        
        weights = [1.5, 1.5, 1.0, 1.0, 1.0, 1.2]
        confidence = self.calculate_confidence(conditions, weights)
        
        return confidence > 0.8

    def detect_h(self, hand_landmarks):
        """
        Detect ASL letter 'H'
        Index and middle fingers extended parallel, others closed
        """
        finger_info = self.get_finger_states(hand_landmarks)
        
        # Check parallel alignment of index and middle fingers
        index_tip = finger_info['index']['tip_pos']
        middle_tip = finger_info['middle']['tip_pos']
        index_pip = finger_info['index']['pip_pos']
        middle_pip = finger_info['middle']['pip_pos']
        
        # Calculate finger alignment
        fingers_parallel = abs((index_tip.x - index_pip.x) - 
                             (middle_tip.x - middle_pip.x)) < 0.05
        
        conditions = [
            finger_info['index']['extended'],      # Index extended
            finger_info['middle']['extended'],     # Middle extended
            not finger_info['ring']['extended'],   # Ring closed
            not finger_info['pinky']['extended'],  # Pinky closed
            not finger_info['thumb']['extended'],  # Thumb tucked
            fingers_parallel,                      # Index and middle parallel
            abs(index_tip.x - middle_tip.x) < 0.1 # Fingers close together
        ]
        
        weights = [1.2, 1.2, 1.0, 1.0, 1.0, 1.5, 1.3]
        confidence = self.calculate_confidence(conditions, weights)
        
        return confidence > 0.8

    def detect_i(self, hand_landmarks):
        """
        Detect ASL letter 'I'
        Pinky extended, all others closed
        """
        finger_info = self.get_finger_states(hand_landmarks)
        
        # Check pinky is straight
        pinky_angle = finger_info['pinky']['angle']
        
        conditions = [
            not finger_info['index']['extended'],   # Index closed
            not finger_info['middle']['extended'],  # Middle closed
            not finger_info['ring']['extended'],    # Ring closed
            finger_info['pinky']['extended'],       # Pinky extended
            not finger_info['thumb']['extended'],   # Thumb tucked
            pinky_angle > 150,                      # Pinky straight
            finger_info['pinky']['tip_pos'].y < 
            finger_info['pinky']['pip_pos'].y       # Pinky pointing up
        ]
        
        weights = [1.0, 1.0, 1.0, 1.5, 1.0, 1.2, 1.3]
        confidence = self.calculate_confidence(conditions, weights)
        
        return confidence > 0.85

    def detect_j(self, hand_landmarks):
        """
        Detect ASL letter 'J'
        Pinky extended, hand rotated, movement downward (simplified static version)
        """
        finger_info = self.get_finger_states(hand_landmarks)
        
        # Similar to 'I' but with rotation
        pinky_tip = finger_info['pinky']['tip_pos']
        pinky_pip = finger_info['pinky']['pip_pos']
        
        # Check if pinky is angled
        pinky_angled = abs(pinky_tip.x - pinky_pip.x) > 0.05
        
        conditions = [
            not finger_info['index']['extended'],   # Index closed
            not finger_info['middle']['extended'],  # Middle closed
            not finger_info['ring']['extended'],    # Ring closed
            finger_info['pinky']['extended'],       # Pinky extended
            not finger_info['thumb']['extended'],   # Thumb tucked
            pinky_angled                           # Pinky angled
        ]
        
        weights = [1.0, 1.0, 1.0, 1.5, 1.0, 1.5]
        confidence = self.calculate_confidence(conditions, weights)
        
        return confidence > 0.8

    def detect_k(self, hand_landmarks):
        """
        Detect ASL letter 'K'
        Index and middle fingers extended upward in V shape, palm forward
        """
        finger_info = self.get_finger_states(hand_landmarks)
        
        # Check V shape between index and middle fingers
        index_tip = finger_info['index']['tip_pos']
        middle_tip = finger_info['middle']['tip_pos']
        index_pip = finger_info['index']['pip_pos']
        
        # Calculate V angle
        v_spread = self.normalized_distance(index_tip, middle_tip, hand_size)
        
        conditions = [
            finger_info['index']['extended'],      # Index extended
            finger_info['middle']['extended'],     # Middle extended
            not finger_info['ring']['extended'],   # Ring closed
            not finger_info['pinky']['extended'],  # Pinky closed
            not finger_info['thumb']['extended'],  # Thumb tucked
            0.1 < v_spread < 0.3,                 # Appropriate V spread
            index_tip.y < index_pip.y             # Fingers pointing up
        ]
        
        weights = [1.2, 1.2, 1.0, 1.0, 1.0, 1.5, 1.1]
        confidence = self.calculate_confidence(conditions, weights)
        
        return confidence > 0.8

    def detect_l(self, hand_landmarks):
        """
        Detect ASL letter 'L'
        Index extended upward, thumb extended to side, forming L shape
        """
        finger_info = self.get_finger_states(hand_landmarks)
        
        # Get positions for L shape check
        thumb_tip = finger_info['thumb']['tip_pos']
        index_tip = finger_info['index']['tip_pos']
        index_pip = finger_info['index']['pip_pos']
        
        # Check L shape angle
        l_angle = self.get_finger_curve_angle(thumb_tip, index_pip, index_tip)
        
        conditions = [
            finger_info['index']['extended'],      # Index extended
            finger_info['thumb']['extended'],      # Thumb extended
            not finger_info['middle']['extended'], # Middle closed
            not finger_info['ring']['extended'],   # Ring closed
            not finger_info['pinky']['extended'],  # Pinky closed
            80 < l_angle < 100,                   # Approximately 90 degree angle
            thumb_tip.x < index_pip.x             # Thumb to side
        ]
        
        weights = [1.3, 1.3, 1.0, 1.0, 1.0, 1.5, 1.2]
        confidence = self.calculate_confidence(conditions, weights)
        
        return confidence > 0.85

    def detect_m(self, hand_landmarks):
        """
        Detect ASL letter 'M'
        Thumb tucked between three folded fingers
        """
        finger_info = self.get_finger_states(hand_landmarks)
        
        # Check positions of folded fingers
        thumb_tip = finger_info['thumb']['tip_pos']
        index_pip = finger_info['index']['pip_pos']
        middle_pip = finger_info['middle']['pip_pos']
        ring_pip = finger_info['ring']['pip_pos']
        
        # Check alignment of folded fingers
        fingers_aligned = (
            abs(index_pip.y - middle_pip.y) < 0.05 and
            abs(middle_pip.y - ring_pip.y) < 0.05
        )
        
        conditions = [
            not finger_info['index']['extended'],   # Index folded
            not finger_info['middle']['extended'],  # Middle folded
            not finger_info['ring']['extended'],    # Ring folded
            not finger_info['pinky']['extended'],   # Pinky closed
            not finger_info['thumb']['extended'],   # Thumb tucked
            fingers_aligned,                        # Three fingers aligned
            thumb_tip.y > index_pip.y              # Thumb below fingers
        ]
        
        weights = [1.2, 1.2, 1.2, 1.0, 1.0, 1.5, 1.3]
        confidence = self.calculate_confidence(conditions, weights)
        
        return confidence > 0.85

    def detect_n(self, hand_landmarks):
        """
        Detect ASL letter 'N'
        Index and middle fingers folded, thumb between them
        """
        finger_info = self.get_finger_states(hand_landmarks)
        
        # Get key positions
        thumb_tip = finger_info['thumb']['tip_pos']
        index_pip = finger_info['index']['pip_pos']
        middle_pip = finger_info['middle']['pip_pos']
        
        # Check thumb position relative to folded fingers
        thumb_between = (
            thumb_tip.x > index_pip.x and
            thumb_tip.x < middle_pip.x and
            thumb_tip.y > min(index_pip.y, middle_pip.y)
        )
        
        conditions = [
            not finger_info['index']['extended'],   # Index folded
            not finger_info['middle']['extended'],  # Middle folded
            not finger_info['ring']['extended'],    # Ring closed
            not finger_info['pinky']['extended'],   # Pinky closed
            not finger_info['thumb']['extended'],   # Thumb tucked
            thumb_between,                          # Thumb between index and middle
            abs(index_pip.y - middle_pip.y) < 0.05 # Fingers aligned
        ]
        
        weights = [1.2, 1.2, 1.0, 1.0, 1.0, 1.5, 1.3]
        confidence = self.calculate_confidence(conditions, weights)
        
        return confidence > 0.85

    def detect_o(self, hand_landmarks):
        """
        Detect ASL letter 'O'
        Fingers curved to form O shape with thumb
        """
        finger_info = self.get_finger_states(hand_landmarks)
        
        # Check O shape formation
        thumb_tip = finger_info['thumb']['tip_pos']
        index_tip = finger_info['index']['tip_pos']
        
        # Calculate O shape metrics
        o_distance = self.normalized_distance(thumb_tip, index_tip, hand_size)
        
        conditions = [
            not any(finger_info[name]['extended']   # No fingers fully extended
                   for name in ['index', 'middle', 'ring', 'pinky']),
            o_distance < 0.1,                       # Thumb and index close
            30 < finger_info['index']['angle'] < 90,# Index curved appropriately
            30 < finger_info['thumb']['angle'] < 90,# Thumb curved appropriately
            abs(thumb_tip.y - index_tip.y) < 0.1   # Thumb and index aligned
        ]
        
        weights = [1.2, 1.5, 1.3, 1.3, 1.2]
        confidence = self.calculate_confidence(conditions, weights)
        
        return confidence > 0.8

    def detect_p(self, hand_landmarks):
        """
        Detect ASL letter 'P'
        Index finger pointing down, thumb and middle finger extended
        """
        finger_info = self.get_finger_states(hand_landmarks)
        
        # Check index pointing down
        index_tip = finger_info['index']['tip_pos']
        index_pip = finger_info['index']['pip_pos']
        pointing_down = index_tip.y > index_pip.y
        
        conditions = [
            finger_info['index']['extended'],      # Index extended
            finger_info['middle']['extended'],     # Middle extended
            not finger_info['ring']['extended'],   # Ring closed
            not finger_info['pinky']['extended'],  # Pinky closed
            finger_info['thumb']['extended'],      # Thumb extended
            pointing_down,                         # Index pointing down
            finger_info['index']['angle'] > 150    # Index straight
        ]
        
        weights = [1.2, 1.2, 1.0, 1.0, 1.2, 1.5, 1.3]
        confidence = self.calculate_confidence(conditions, weights)
        
        return confidence > 0.85

    def detect_q(self, hand_landmarks):
        """
        Detect ASL letter 'Q'
        Index and thumb form downward pointing Q
        """
        finger_info = self.get_finger_states(hand_landmarks)
        
        # Check downward pointing
        index_tip = finger_info['index']['tip_pos']
        index_pip = finger_info['index']['pip_pos']
        pointing_down = index_tip.y > index_pip.y
        
        conditions = [
            finger_info['index']['extended'],      # Index extended
            not finger_info['middle']['extended'], # Middle closed
            not finger_info['ring']['extended'],   # Ring closed
            not finger_info['pinky']['extended'],  # Pinky closed
            finger_info['thumb']['extended'],      # Thumb extended
            pointing_down,                         # Index pointing down
            finger_info['index']['angle'] > 150    # Index straight
        ]
        
        weights = [1.3, 1.0, 1.0, 1.0, 1.2, 1.5, 1.3]
        confidence = self.calculate_confidence(conditions, weights)
        
        return confidence > 0.85

    def detect_r(self, hand_landmarks):
        """
        Detect ASL letter 'R'
        Index and middle fingers crossed
        """
        finger_info = self.get_finger_states(hand_landmarks)
        
        # Check finger crossing
        index_tip = finger_info['index']['tip_pos']
        middle_tip = finger_info['middle']['tip_pos']
        index_pip = finger_info['index']['pip_pos']
        middle_pip = finger_info['middle']['pip_pos']
        
        # Calculate crossing
        fingers_crossed = (
            abs(index_tip.x - middle_tip.x) < 0.05 and
            abs(index_pip.x - middle_pip.x) > 0.05
        )
        
        conditions = [
            finger_info['index']['extended'],      # Index extended
            finger_info['middle']['extended'],     # Middle extended
            not finger_info['ring']['extended'],   # Ring closed
            not finger_info['pinky']['extended'],  # Pinky closed
            not finger_info['thumb']['extended'],  # Thumb tucked
            fingers_crossed,                       # Fingers crossed
            index_tip.y < index_pip.y             # Fingers pointing up
        ]
        
        weights = [1.2, 1.2, 1.0, 1.0, 1.0, 1.5, 1.2]
        confidence = self.calculate_confidence(conditions, weights)
        
        return confidence > 0.85

    def detect_s(self, hand_landmarks):
        """
        Detect ASL letter 'S'
        Fist with thumb wrapped around front
        """
        finger_info = self.get_finger_states(hand_landmarks)
        
        # Check thumb position
        thumb_tip = finger_info['thumb']['tip_pos']
        index_pip = finger_info['index']['pip_pos']
        
        thumb_front = thumb_tip.z < index_pip.z
        
        conditions = [
            not finger_info['index']['extended'],   # Index closed
            not finger_info['middle']['extended'],  # Middle closed
            not finger_info['ring']['extended'],    # Ring closed
            not finger_info['pinky']['extended'],   # Pinky closed
            not finger_info['thumb']['extended'],   # Thumb wrapped
            thumb_front,                            # Thumb in front
            all(finger_info[name]['angle'] < 45     # Fingers tightly curled
                for name in ['index', 'middle', 'ring', 'pinky'])
        ]
        
        weights = [1.0, 1.0, 1.0, 1.0, 1.2, 1.5, 1.3]
        confidence = self.calculate_confidence(conditions, weights)
        
        return confidence > 0.85



    def detect_t(self, hand_landmarks):
        """
        Detect ASL letter 'T'
        Thumb and index extended, pointing up
        """
        finger_info = self.get_finger_states(hand_landmarks)

        # Check index pointing up
        index_tip = finger_info['index']['tip_pos']
        index_pip = finger_info['index']['pip_pos']
        pointing_up = index_tip.y < index_pip.y

        conditions = [
            finger_info['index']['extended'],      # Index extended
            not finger_info['middle']['extended'], # Middle closed
            not finger_info['ring']['extended'],   # Ring closed
            not finger_info['pinky']['extended'],  # Pinky closed
            finger_info['thumb']['extended'],      # Thumb extended
            pointing_up,                           # Index pointing up
            finger_info['index']['angle'] > 150    # Index straight
        ]

        weights = [1.2, 1.0, 1.0, 1.0, 1.2, 1.5, 1.3]
        confidence = self.calculate_confidence(conditions, weights)

        return confidence > 0.85

    def detect_u(self, hand_landmarks):
        """
        Detect ASL letter 'U'
        Index and middle fingers extended together, pointing up
        """
        finger_info = self.get_finger_states(hand_landmarks)
        
        # Check fingers pointing up and parallel
        index_tip = finger_info['index']['tip_pos']
        middle_tip = finger_info['middle']['tip_pos']
        index_pip = finger_info['index']['pip_pos']
        middle_pip = finger_info['middle']['pip_pos']
        
        # Add this missing definition
        pointing_up = (index_tip.y < index_pip.y and 
            middle_tip.y < middle_pip.y)
        
        fingers_parallel = abs(index_tip.x - middle_tip.x) == abs(index_pip.x - middle_pip.x)
        
        conditions = [
            finger_info['index']['extended'],      # Index extended
            finger_info['middle']['extended'],     # Middle extended
            not finger_info['ring']['extended'],   # Ring closed
            not finger_info['pinky']['extended'],  # Pinky closed
            not finger_info['thumb']['extended'],  # Thumb tucked
            fingers_parallel,                      # Fingers parallel
            pointing_up                           # Fingers pointing up
        ]
        
        weights = [1.2, 1.2, 1.0, 1.0, 1.0, 1.5, 1.3]
        confidence = self.calculate_confidence(conditions, weights)
        
        return confidence > 0.85
    
    def detect_v(self, hand_landmarks):
        """
        Detect ASL letter 'V'
        Index and middle fingers extended in V shape
        """
        finger_info = self.get_finger_states(hand_landmarks)
        
        # Check V shape formation
        index_tip = finger_info['index']['tip_pos']
        middle_tip = finger_info['middle']['tip_pos']
        index_pip = finger_info['index']['pip_pos']
        middle_pip = finger_info['middle']['pip_pos']
    
        v_shape = abs(index_tip.x - middle_tip.x) > abs(index_pip.x - middle_pip.x)
        pointing_up = index_tip.y < index_pip.y and middle_tip.y < middle_pip.y
        
        conditions = [
            finger_info['index']['extended'],      # Index extended
            finger_info['middle']['extended'],     # Middle extended
            not finger_info['ring']['extended'],   # Ring closed
            not finger_info['pinky']['extended'],  # Pinky closed
            not finger_info['thumb']['extended'],  # Thumb tucked
            v_shape,                              # Fingers in V shape
            pointing_up                           # Fingers pointing up
        ]
        
        weights = [1.2, 1.2, 1.0, 1.0, 1.0, 1.5, 1.3]
        confidence = self.calculate_confidence(conditions, weights)
        
        return confidence > 0.85
    
    def detect_w(self, hand_landmarks):
        """
        Detect ASL letter 'W'
        Index, middle, and ring fingers extended in W shape
        """
        finger_info = self.get_finger_states(hand_landmarks)
        
        # Check fingers pointing up
        index_tip = finger_info['index']['tip_pos']
        index_pip = finger_info['index']['pip_pos']
        pointing_up = index_tip.y < index_pip.y
        
        conditions = [
            finger_info['index']['extended'],      # Index extended
            finger_info['middle']['extended'],     # Middle extended
            finger_info['ring']['extended'],       # Ring extended
            not finger_info['pinky']['extended'],  # Pinky closed
            not finger_info['thumb']['extended'],  # Thumb tucked
            pointing_up,                          # Fingers pointing up
            finger_info['index']['angle'] > 150   # Fingers straight
        ]
        
        weights = [1.2, 1.2, 1.2, 1.0, 1.0, 1.5, 1.3]
        confidence = self.calculate_confidence(conditions, weights)
        
        return confidence > 0.85
    
    def detect_x(self, hand_landmarks):
        """
        Detect ASL letter 'X'
        Index finger bent at first joint, pointing down
        """
        finger_info = self.get_finger_states(hand_landmarks)
        
        # Check index pointing down and bent
        index_tip = finger_info['index']['tip_pos']
        index_pip = finger_info['index']['pip_pos']
        pointing_down = index_tip.y > index_pip.y
        
        conditions = [
            finger_info['index']['extended'],      # Index partially extended
            not finger_info['middle']['extended'], # Middle closed
            not finger_info['ring']['extended'],   # Ring closed
            not finger_info['pinky']['extended'],  # Pinky closed
            not finger_info['thumb']['extended'],  # Thumb tucked
            pointing_down,                        # Index pointing down
            45 < finger_info['index']['angle'] < 90  # Index bent
        ]
        
        weights = [1.2, 1.0, 1.0, 1.0, 1.0, 1.5, 1.3]
        confidence = self.calculate_confidence(conditions, weights)
        
        return confidence > 0.85
    
    def detect_y(self, hand_landmarks):
        """
        Detect ASL letter 'Y'
        Thumb and pinky extended, other fingers closed
        """
        finger_info = self.get_finger_states(hand_landmarks)
        
        conditions = [
            not finger_info['index']['extended'],  # Index closed
            not finger_info['middle']['extended'], # Middle closed
            not finger_info['ring']['extended'],   # Ring closed
            finger_info['pinky']['extended'],      # Pinky extended
            finger_info['thumb']['extended'],      # Thumb extended
            finger_info['pinky']['angle'] > 150,   # Pinky straight
            finger_info['thumb']['angle'] > 150    # Thumb straight
        ]
        
        weights = [1.0, 1.0, 1.0, 1.3, 1.3, 1.2, 1.2]
        confidence = self.calculate_confidence(conditions, weights)
        
        return confidence > 0.85
    
    def detect_z(self, hand_landmarks):
        """
        Detect ASL letter 'Z'
        Index finger drawing Z shape in air
        Note: This might need additional motion tracking logic for accurate detection
        """
        finger_info = self.get_finger_states(hand_landmarks)
        
        # Basic position check (though Z requires motion tracking for full accuracy)
        index_tip = finger_info['index']['tip_pos']
        index_pip = finger_info['index']['pip_pos']
        
        conditions = [
            finger_info['index']['extended'],      # Index extended
            not finger_info['middle']['extended'], # Middle closed
            not finger_info['ring']['extended'],   # Ring closed
            not finger_info['pinky']['extended'],  # Pinky closed
            not finger_info['thumb']['extended'],  # Thumb tucked
            finger_info['index']['angle'] > 150,   # Index straight
            index_tip.y < index_pip.y             # Index pointing up
        ]
        
        weights = [1.3, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2]
        confidence = self.calculate_confidence(conditions, weights)
        
        return confidence > 0.85

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
            print("Error: Can't receive frame from camera")
            break
            
        processed_frame = detector.process_frame(frame)
        cv2.imshow('ASL Fingerspelling Detection', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()