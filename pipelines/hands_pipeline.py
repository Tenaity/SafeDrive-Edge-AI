import mediapipe as mp
import cv2
import time

class HandsPipeline:
    """
    HandsPipeline:
    - Detect hands using MediaPipe Hands
    - Warn if no hand visible for N seconds
    """

    def __init__(self, no_hand_time=3.0):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.no_hand_time = no_hand_time
        self.no_hand_since = None

    def run(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        hands_present = res.multi_hand_landmarks is not None

        warning = False
        if not hands_present:
            if self.no_hand_since is None:
                self.no_hand_since = time.time()
            elif time.time() - self.no_hand_since >= self.no_hand_time:
                warning = True
        else:
            self.no_hand_since = None

        return {
            "hands_present": hands_present,
            "hands_warning": warning
        }
