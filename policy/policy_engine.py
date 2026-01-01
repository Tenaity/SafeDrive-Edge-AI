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

    def decide(self, vision, driver, hands, crane):
        """
        Decide alert level based on all inputs.
        Anti-spam logic: If crane is NOT lifting, suppress alerts.
        """
        is_lifting = crane.get("is_lifting", False)

        # If not lifting, we suppress alerts (Anti-spam)
        # Note: If you want to ALWAYS alert on Drowsiness even when stopped, remove this check for drowsiness.
        # Based on user request "while free, do not warn", we return NONE.
        if not is_lifting:
            return AlertLevel.NONE

        if driver.get("drowsy") or driver.get("distracted"):
            return AlertLevel.HIGH

        if vision.get("phone") or hands.get("hands_warning"):
            return AlertLevel.MEDIUM

        return AlertLevel.NONE
