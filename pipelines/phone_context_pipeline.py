class PhoneContextPipeline:
    """
    PhoneContextPipeline:
    - Analyze phone candidates in seated-driver context
    - No screen-brightness logic
    - Only provide geometric context:
      lap zone, mid-leg zone, head_down, lap_attention
    """

    def __init__(self):
        pass

    @staticmethod
    def _box_center(box):
        if not box or len(box) != 4:
            return None
        x1, y1, x2, y2 = map(float, box)
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _lap_zone_box(self, person_box):
        if person_box is None:
            return None

        px1, py1, px2, py2 = map(float, person_box)
        pw = px2 - px1
        ph = py2 - py1

        x1 = px1 - pw * 0.05
        x2 = px2 + pw * 0.05
        y1 = py1 + ph * 0.45
        y2 = py1 + ph * 1.05

        return [x1, y1, x2, y2]

    def _mid_leg_zone_box(self, person_box):
        if person_box is None:
            return None

        px1, py1, px2, py2 = map(float, person_box)
        pw = px2 - px1
        ph = py2 - py1

        x1 = px1 + pw * 0.24
        x2 = px2 - pw * 0.24
        y1 = py1 + ph * 0.52
        y2 = py1 + ph * 1.08

        return [x1, y1, x2, y2]

    def _point_in_box(self, pt, box):
        if pt is None or box is None:
            return False
        x, y = pt
        x1, y1, x2, y2 = map(float, box)
        return x1 <= x <= x2 and y1 <= y <= y2

    def _box_in_zone(self, phone_box, zone_box):
        c = self._box_center(phone_box)
        return self._point_in_box(c, zone_box)

    def run(self, frame, person_box, phone_candidates, hand_centers, face_center, head_down):
        lap_zone_box = self._lap_zone_box(person_box)
        mid_leg_zone_box = self._mid_leg_zone_box(person_box)

        phone_contexts = []

        for pb in phone_candidates or []:
            in_lap_zone = self._box_in_zone(pb, lap_zone_box)
            in_mid_leg_zone = self._box_in_zone(pb, mid_leg_zone_box)

            phone_contexts.append({
                "box": pb,
                "in_lap_zone": in_lap_zone,
                "in_mid_leg_zone": in_mid_leg_zone,
                "source": "yolo_phone"
            })

        lap_attention = False
        if head_down and hand_centers and lap_zone_box is not None:
            for hc in hand_centers:
                if self._point_in_box(hc, lap_zone_box):
                    lap_attention = True
                    break

        return {
            "phone_contexts": phone_contexts,
            "lap_zone_box": lap_zone_box,
            "mid_leg_zone_box": mid_leg_zone_box,
            "lap_attention": lap_attention,
            "head_down": bool(head_down),
        }