from enum import Enum


class AlertLevel(Enum):
    NONE = 0
    PHONE = 1
    MEDIUM = 2
    HIGH = 3


class PolicyEngine:
    """
    Priority:
    HIGH   = drowsy OR yawning OR distracted
    PHONE  = confirmed phone usage
    MEDIUM = no hand
    NONE   = normal
    """

    def decide(self, vision, driver, hands, crane, phone_usage=None):
        vision = vision or {}
        driver = driver or {}
        hands = hands or {}
        crane = crane or {}
        phone_usage = phone_usage or {}

        if driver.get("drowsy") or driver.get("yawning") or driver.get("distracted"):
            return AlertLevel.HIGH

        if phone_usage.get("phone_using"):
            return AlertLevel.PHONE

        if hands.get("no_hand"):
            return AlertLevel.MEDIUM

        return AlertLevel.NONE