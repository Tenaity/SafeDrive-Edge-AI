from enum import Enum

class AlertLevel(Enum):
    NONE = 0
    MEDIUM = 1
    HIGH = 2

class PolicyEngine:
    """
    PolicyEngine:
    - MEDIUM: phone OR hands off
    - HIGH: drowsy OR distracted
    """

    def decide(self, vision, driver, hands):
        if driver.get("drowsy") or driver.get("distracted"):
            return AlertLevel.HIGH

        if vision.get("phone") or hands.get("hands_warning"):
            return AlertLevel.MEDIUM

        return AlertLevel.NONE
