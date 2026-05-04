import math


def safe_box(box) -> list[float] | None:
    """Validate and normalize a bounding box [x1, y1, x2, y2]. Returns None if invalid."""
    if box is None:
        return None
    try:
        if len(box) != 4:
            return None
        x1, y1, x2, y2 = map(float, box)
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def box_center(box) -> tuple[float, float] | None:
    """Return the (cx, cy) center of a bounding box, or None if invalid."""
    b = safe_box(box)
    if b is None:
        return None
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_size(box) -> tuple[float, float]:
    """Return (width, height) of a bounding box; (0.0, 0.0) if invalid."""
    b = safe_box(box)
    if b is None:
        return 0.0, 0.0
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def box_area(box) -> float:
    """Return the area of a bounding box; 0.0 if invalid."""
    w, h = box_size(box)
    return w * h


def intersection_area(box_a, box_b) -> float:
    """Return the intersection area of two bounding boxes."""
    a = safe_box(box_a)
    b = safe_box(box_b)
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def overlap_ratio(box_a, box_b) -> float:
    """Return what fraction of box_a is covered by box_b (0.0–1.0)."""
    area_a = box_area(box_a)
    if area_a <= 1e-6:
        return 0.0
    return intersection_area(box_a, box_b) / area_a


def point_distance(p1, p2) -> float:
    """Euclidean distance between two (x, y) points. Returns 1e9 if either is None."""
    if p1 is None or p2 is None:
        return 1e9
    return math.hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1]))


def expand_box(box, sx: float = 0.0, sy: float = 0.0) -> list[float] | None:
    """Expand a box by sx fraction horizontally and sy fraction vertically on each side."""
    b = safe_box(box)
    if b is None:
        return None
    x1, y1, x2, y2 = b
    w = x2 - x1
    h = y2 - y1
    return [x1 - w * sx, y1 - h * sy, x2 + w * sx, y2 + h * sy]


def point_in_box(point, box) -> bool:
    """Return True if point (x, y) lies within the bounding box."""
    if point is None:
        return False
    b = safe_box(box)
    if b is None:
        return False
    x, y = float(point[0]), float(point[1])
    x1, y1, x2, y2 = b
    return x1 <= x <= x2 and y1 <= y <= y2
