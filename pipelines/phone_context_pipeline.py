import numpy as np

from utils.box_utils import safe_box, box_center, box_size, box_area, intersection_area, overlap_ratio, point_distance, point_in_box
from utils.types import PhoneContextOut


class PhoneContextPipeline:
    """
    PhoneContextPipeline:
    - Analyze phone candidates in seated-driver context
    - person_box ở đây nên là driver_body_box mở rộng xuống thân dưới
    - giữ mạnh vùng đùi / giữa chân để không mất phone thấp
    - nới vùng hai bên để đỡ hụt khi camera lệch trái/phải
    - bổ sung overlap ratio để cứu case bbox lệch tâm / phone đặt ngang
    """

    def __init__(self):
        pass

    def _lap_zone_box(self, person_box):
        b = safe_box(person_box)
        if b is None:
            return None
        px1, py1, px2, py2 = b
        pw = px2 - px1
        ph = py2 - py1
        return [px1 - pw * 0.10, py1 + ph * 0.40, px2 + pw * 0.10, py1 + ph * 1.14]

    def _mid_leg_zone_box(self, person_box):
        b = safe_box(person_box)
        if b is None:
            return None
        px1, py1, px2, py2 = b
        pw = px2 - px1
        ph = py2 - py1
        return [px1 + pw * 0.08, py1 + ph * 0.46, px2 - pw * 0.08, py1 + ph * 1.22]

    def _side_leg_zone_boxes(self, person_box):
        """Hai vùng đùi lệch trái/phải để cứu case camera lệch và phone nằm ở một bên."""
        b = safe_box(person_box)
        if b is None:
            return None, None
        px1, py1, px2, py2 = b
        pw = px2 - px1
        ph = py2 - py1
        left_box = [px1 - pw * 0.08, py1 + ph * 0.45, px1 + pw * 0.44, py1 + ph * 1.18]
        right_box = [px2 - pw * 0.44, py1 + ph * 0.45, px2 + pw * 0.08, py1 + ph * 1.18]
        return left_box, right_box

    def _max_hand_distance(self, person_box):
        b = safe_box(person_box)
        if b is None:
            return 0.0
        px1, py1, px2, py2 = b
        return max(36.0, 0.22 * ((px2 - px1) + (py2 - py1)))

    def _zone_attention_strength(self, hand_centers, zone_box, person_box):
        if not hand_centers or zone_box is None:
            return 0.0
        limit = self._max_hand_distance(person_box)
        best = 0.0
        for hc in hand_centers:
            if point_in_box(hc, zone_box):
                best = 1.0
                continue
            c = box_center(zone_box)
            d = point_distance(hc, c)
            if d <= limit:
                best = max(best, max(0.0, 1.0 - d / max(1.0, limit)))
        return best

    def run(self, frame: np.ndarray, person_box, phone_candidates, hand_centers, face_center, head_down) -> PhoneContextOut:
        lap_zone_box = self._lap_zone_box(person_box)
        mid_leg_zone_box = self._mid_leg_zone_box(person_box)
        left_leg_zone_box, right_leg_zone_box = self._side_leg_zone_boxes(person_box)

        phone_contexts = []

        for pb in phone_candidates or []:
            lap_overlap = overlap_ratio(pb, lap_zone_box)
            mid_leg_overlap = overlap_ratio(pb, mid_leg_zone_box)
            left_leg_overlap = overlap_ratio(pb, left_leg_zone_box)
            right_leg_overlap = overlap_ratio(pb, right_leg_zone_box)

            in_lap_zone = point_in_box(box_center(pb), lap_zone_box) or lap_overlap >= 0.18
            in_mid_leg_zone = point_in_box(box_center(pb), mid_leg_zone_box) or mid_leg_overlap >= 0.18
            in_left_leg_zone = point_in_box(box_center(pb), left_leg_zone_box) or left_leg_overlap >= 0.18
            in_right_leg_zone = point_in_box(box_center(pb), right_leg_zone_box) or right_leg_overlap >= 0.18

            phone_contexts.append({
                "box": pb,
                "in_lap_zone": bool(in_lap_zone),
                "in_mid_leg_zone": bool(in_mid_leg_zone),
                "in_left_leg_zone": bool(in_left_leg_zone),
                "in_right_leg_zone": bool(in_right_leg_zone),
                "lap_overlap": round(float(lap_overlap), 4),
                "mid_leg_overlap": round(float(mid_leg_overlap), 4),
                "left_leg_overlap": round(float(left_leg_overlap), 4),
                "right_leg_overlap": round(float(right_leg_overlap), 4),
                "source": "yolo_phone",
            })

        lap_attention_strength = self._zone_attention_strength(hand_centers, lap_zone_box, person_box)
        left_side_attention_strength = self._zone_attention_strength(hand_centers, left_leg_zone_box, person_box)
        right_side_attention_strength = self._zone_attention_strength(hand_centers, right_leg_zone_box, person_box)

        return {
            "phone_contexts": phone_contexts,
            "lap_zone_box": lap_zone_box,
            "mid_leg_zone_box": mid_leg_zone_box,
            "left_leg_zone_box": left_leg_zone_box,
            "right_leg_zone_box": right_leg_zone_box,
            "lap_attention": lap_attention_strength >= 0.35,
            "left_side_attention": left_side_attention_strength >= 0.35,
            "right_side_attention": right_side_attention_strength >= 0.35,
            "lap_attention_strength": round(float(lap_attention_strength), 4),
            "left_side_attention_strength": round(float(left_side_attention_strength), 4),
            "right_side_attention_strength": round(float(right_side_attention_strength), 4),
            "head_down": bool(head_down),
            "body_box": person_box,
            "face_center": face_center,
        }