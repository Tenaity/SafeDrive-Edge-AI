from enum import Enum

class AlertLevel(Enum):
    NONE = 0
    PHONE = 1
    MEDIUM = 2
    HIGH = 3

class PolicyEngine:
    """
    PolicyEngine:
    - MEDIUM: phone OR hands off
    - HIGH: drowsy OR distracted
    """

    def decide(self, vision, driver, hands, crane):
        if driver.get("drowsy") or driver.get("distracted"):
            return AlertLevel.HIGH

        if vision.get("phone"):
            return AlertLevel.PHONE
        
        if hands.get("hands_warning"):
            return AlertLevel.MEDIUM

        return AlertLevel.NONE
