import math


def euclid(p1, p2) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def landmark_to_px(landmark, w: int, h: int) -> tuple[float, float]:
    return (landmark.x * w, landmark.y * h)
