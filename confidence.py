import cv2
import mediapipe as mp
import math

class ASLFingerSpellingDetector:
    def detect_eco(self, hand_landmarks):
        # Get key points
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        thumb_mcp = hand_landmarks.landmark[2]
        index_tip = hand_landmarks.landmark[8]
        index_pip = hand_landmarks.landmark[6]
        index_mcp = hand_landmarks.landmark[5]
        middle_tip = hand_landmarks.landmark[12]
        middle_pip = hand_landmarks.landmark[10]
        middle_mcp = hand_landmarks.landmark[9]
        ring_tip = hand_landmarks.landmark[16]
        ring_pip = hand_landmarks.landmark[14]
        pinky_tip = hand_landmarks.landmark[20]
        pinky_pip = hand_landmarks.landmark[18]
        wrist = hand_landmarks.landmark[0]

        # Detect E
        e_confidence = self.detect_e_confidence(thumb_tip, thumb_mcp, index_tip, index_pip, index_mcp,
                                                middle_tip, middle_pip, ring_tip, ring_pip, pinky_tip, pinky_pip)

        # Detect C
        c_confidence = self.detect_c_confidence(thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip,
                                                index_pip, middle_pip, ring_pip, pinky_pip)

        # Detect O
        o_confidence = self.detect_o_confidence(thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip,
                                                middle_pip, ring_pip, pinky_pip, wrist)

        # Determine the highest confidence
        confidences = {'E': e_confidence, 'C': c_confidence, 'O': o_confidence}
        detected_letter = max(confidences, key=confidences.get)
        highest_confidence = confidences[detected_letter]

        # Debug information
        print("\nDetection Results:")
        print(f"E Confidence: {e_confidence:.2f}")
        print(f"C Confidence: {c_confidence:.2f}")
        print(f"O Confidence: {o_confidence:.2f}")
        print(f"Detected Letter: {detected_letter} with confidence {highest_confidence:.2f}")

        return detected_letter, highest_confidence

    def detect_e_confidence(self, thumb_tip, thumb_mcp, index_tip, index_pip, index_mcp,
                            middle_tip, middle_pip, ring_tip, ring_pip, pinky_tip, pinky_pip):
        fingers_curled = all([
            index_tip.y > index_pip.y,
            middle_tip.y > middle_pip.y,
            ring_tip.y > ring_pip.y,
            pinky_tip.y > pinky_pip.y
        ])
        thumb_across = (thumb_tip.y >= middle_tip.y and thumb_tip.x < middle_pip.x)
        thumb_in_front = thumb_tip.z < index_pip.z
        fingers_together = all([
            abs(index_tip.x - middle_tip.x) < 0.08,
            abs(middle_tip.x - ring_tip.x) < 0.08,
            abs(ring_tip.x - pinky_tip.x) < 0.08
        ])
        thumb_position = (thumb_tip.x < index_mcp.x and thumb_tip.y > thumb_mcp.y)

        conditions = [fingers_curled, thumb_across, thumb_in_front, fingers_together, thumb_position]
        weights = [1.5, 1.4, 1.2, 1.1, 1.3]
        return self.calculate_confidence(conditions, weights)

    def detect_c_confidence(self, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip,
                            index_pip, middle_pip, ring_pip, pinky_pip):
        thumb_index_curve = (abs(thumb_tip.y - index_tip.y) < 0.1 and
                             thumb_tip.x < index_tip.x)
        fingers_curved = all([
            index_tip.y > index_pip.y,
            middle_tip.y > middle_pip.y,
            ring_tip.y > ring_pip.y,
            pinky_tip.y > pinky_pip.y
        ])
        fingers_aligned = (max(index_tip.x, middle_tip.x, ring_tip.x, pinky_tip.x) -
                           min(index_tip.x, middle_tip.x, ring_tip.x, pinky_tip.x) < 0.15)
        c_shape = (thumb_tip.y < pinky_tip.y
                   and thumb_tip.x < index_pip.x < middle_tip.x < ring_tip.x < pinky_tip.x) 
        palm_facing = wrist.z > middle_mcp.z

        conditions = [thumb_index_curve, fingers_curved, fingers_aligned, c_shape]
        weights = [1.3, 1.2, 1.1, 1.4]
        return self.calculate_confidence(conditions, weights)

    def detect_o_confidence(self, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip,
                            middle_pip, ring_pip, pinky_pip, wrist):
        thumb_index_circle = 0.02 < self.calculate_distance(thumb_tip, index_tip) < 0.1
        other_fingers_extended = all([
            middle_tip.y < middle_pip.y,
            ring_tip.y < ring_pip.y,
            pinky_tip.y < pinky_pip.y
        ])
        fingers_close_together = all([
            0.02 < abs(middle_tip.x - ring_tip.x) < 0.08,
            0.02 < abs(ring_tip.x - pinky_tip.x) < 0.08
        ])
        palm_facing_camera = middle_pip.z < wrist.z

        conditions = [thumb_index_circle, other_fingers_extended, fingers_close_together, palm_facing_camera]
        weights = [1.5, 1.2, 1.0, 1.3]
        return self.calculate_confidence(conditions, weights)

    def calculate_confidence(self, conditions, weights):
        weighted_sum = sum(weight * (1 if condition else 0) for weight, condition in zip(weights, conditions))
        total_weight = sum(weights)
        return weighted_sum / total_weight if total_weight > 0 else 0


    def calculate_distance(self, point1, point2):
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)

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