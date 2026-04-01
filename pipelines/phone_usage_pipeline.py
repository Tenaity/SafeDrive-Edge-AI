import math
from collections import deque


class PhoneUsagePipeline:
    """
    Strong phone-usage reasoning for seated driver:
    - NO screen brightness logic
    - Use geometry + body interaction only
    - supports phone near face
    - supports phone on lap / between legs
    - supports thigh as a strong cue similar to hand
    """

    def __init__(self):
        self.confirm_frames_on = 5
        self.confirm_frames_off = 3

        self.use_counter = 0
        self.clear_counter = 0
        self.phone_using_state = False

        self.phone_center_history = deque(maxlen=10)

        self.wrist_phone_dist_ratio = 0.52
        self.elbow_phone_dist_ratio = 0.46
        self.face_phone_dist_ratio = 0.45
        self.thigh_phone_dist_ratio = 0.35

        self.use_score_threshold = 6

    @staticmethod
    def _box_center(box):
        if not box or len(box) != 4:
            return None
        x1, y1, x2, y2 = map(float, box)
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _distance(p1, p2):
        if p1 is None or p2 is None:
            return 1e9
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _phone_in_driver_zone(self, phone_box, person_box):
        if phone_box is None or person_box is None:
            return False

        pc = self._box_center(phone_box)
        if pc is None:
            return False

        px1, py1, px2, py2 = map(float, person_box)
        pw = px2 - px1
        ph = py2 - py1

        zx1 = px1 - pw * 0.18
        zx2 = px2 + pw * 0.18
        zy1 = py1 - ph * 0.10
        zy2 = py1 + ph * 1.05

        return zx1 <= pc[0] <= zx2 and zy1 <= pc[1] <= zy2

    def _phone_near_wrist(self, phone_box, wrist_points, person_box):
        if phone_box is None or person_box is None or not wrist_points:
            return False

        phone_c = self._box_center(phone_box)
        px1, py1, px2, py2 = map(float, person_box)
        pw = px2 - px1
        limit = pw * self.wrist_phone_dist_ratio

        for wp in wrist_points:
            if wp is not None and self._distance(phone_c, wp) <= limit:
                return True
        return False

    def _phone_near_elbow(self, phone_box, elbow_points, person_box):
        if phone_box is None or person_box is None or not elbow_points:
            return False

        phone_c = self._box_center(phone_box)
        px1, py1, px2, py2 = map(float, person_box)
        pw = px2 - px1
        limit = pw * self.elbow_phone_dist_ratio

        for ep in elbow_points:
            if ep is not None and self._distance(phone_c, ep) <= limit:
                return True
        return False

    def _phone_near_face(self, phone_box, face_center, person_box):
        if phone_box is None or face_center is None or person_box is None:
            return False

        phone_c = self._box_center(phone_box)
        px1, py1, px2, py2 = map(float, person_box)
        pw = px2 - px1
        limit = pw * self.face_phone_dist_ratio
        return self._distance(phone_c, face_center) <= limit

    def _phone_near_thigh(self, phone_box, thigh_points, person_box):
        if phone_box is None or person_box is None or not thigh_points:
            return False

        phone_c = self._box_center(phone_box)
        px1, py1, px2, py2 = map(float, person_box)
        pw = px2 - px1
        limit = pw * self.thigh_phone_dist_ratio

        for tp in thigh_points:
            if tp is not None and self._distance(phone_c, tp) <= limit:
                return True
        return False

    def _temporal_box_stable(self, phone_box):
        c = self._box_center(phone_box)
        if c is None:
            self.phone_center_history.clear()
            return False

        self.phone_center_history.append(c)

        if len(self.phone_center_history) < 3:
            return False

        xs = [p[0] for p in self.phone_center_history]
        ys = [p[1] for p in self.phone_center_history]

        if (max(xs) - min(xs)) > 320:
            return False
        if (max(ys) - min(ys)) > 320:
            return False

        return True

    def _score_phone(
        self,
        phone_box,
        person_box,
        wrist_points,
        elbow_points,
        thigh_points,
        face_center,
        phone_ctx,
        context_out,
    ):
        score = 0

        in_driver_zone = self._phone_in_driver_zone(phone_box, person_box)
        near_wrist = self._phone_near_wrist(phone_box, wrist_points, person_box)
        near_elbow = self._phone_near_elbow(phone_box, elbow_points, person_box)
        near_face = self._phone_near_face(phone_box, face_center, person_box)
        near_thigh = self._phone_near_thigh(phone_box, thigh_points, person_box)
        temporal_ok = self._temporal_box_stable(phone_box)

        in_lap_zone = phone_ctx.get("in_lap_zone", False)
        in_mid_leg_zone = phone_ctx.get("in_mid_leg_zone", False)

        head_down = context_out.get("head_down", False)
        lap_attention = context_out.get("lap_attention", False)

        if in_driver_zone:
            score += 2

        if near_wrist:
            score += 3

        if near_elbow:
            score += 1

        if near_face:
            score += 2

        if near_thigh:
            score += 3

        if in_lap_zone:
            score += 2

        if in_mid_leg_zone:
            score += 2

        if temporal_ok:
            score += 2

        if head_down:
            score += 1

        if lap_attention:
            score += 2

        # đùi giống như tay
        if near_thigh and temporal_ok:
            score += 2

        # trên đùi / giữa hai chân và đang nhìn xuống
        if (in_lap_zone or in_mid_leg_zone) and head_down:
            score += 2

        # trên đùi / giữa hai chân và có tay/đùi liên hệ
        if (in_lap_zone or in_mid_leg_zone) and (near_thigh or near_wrist or near_elbow):
            score += 3

        # trường hợp bấm trên đùi nhưng không cầm rõ
        if (in_lap_zone or in_mid_leg_zone) and lap_attention and temporal_ok:
            score += 3

        return score

    def run(
        self,
        person_box,
        phone_candidates,
        hand_centers,
        face_center,
        context_out=None,
        left_wrist=None,
        right_wrist=None,
        left_elbow=None,
        right_elbow=None,
        left_shoulder=None,
        right_shoulder=None,
        left_thigh_center=None,
        right_thigh_center=None,
    ):
        context_out = context_out or {}
        phone_contexts = context_out.get("phone_contexts", [])

        wrist_points = [left_wrist, right_wrist]
        elbow_points = [left_elbow, right_elbow]
        thigh_points = [left_thigh_center, right_thigh_center]

        best_phone_box = None
        best_score = -1
        best_source = None

        for ctx in phone_contexts:
            pb = ctx.get("box")
            score = self._score_phone(
                pb,
                person_box,
                wrist_points,
                elbow_points,
                thigh_points,
                face_center,
                ctx,
                context_out
            )
            if score > best_score:
                best_score = score
                best_phone_box = pb
                best_source = ctx.get("source")

        using_now = best_score >= self.use_score_threshold

        if using_now:
            self.use_counter += 1
            self.clear_counter = 0
            if self.use_counter >= self.confirm_frames_on:
                self.phone_using_state = True
        else:
            self.clear_counter += 1
            self.use_counter = 0
            if self.clear_counter >= self.confirm_frames_off:
                self.phone_using_state = False

        if not using_now and best_phone_box is None:
            self.phone_center_history.clear()

        return {
            "phone_using": self.phone_using_state,
            "best_phone_box": best_phone_box,
            "score": best_score,
            "source": best_source,
        }