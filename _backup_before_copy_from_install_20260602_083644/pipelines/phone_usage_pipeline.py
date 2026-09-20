from collections import deque

from utils.box_utils import safe_box, box_center, box_size, box_area, intersection_area, point_distance
from utils.types import PhoneUsageOut


class PhoneUsagePipeline:
    """
    Strong phone-usage reasoning for seated driver:
    - Dùng person_box ở đây như driver_body_box
    - Giữ phone thấp / trên đùi / giữa hai chân
    - Giảm phụ thuộc vào việc phải nhìn thấy đúng tay bên đó
    - Tăng cứu case phone nằm ngang / camera lệch trái phải
    - Chống nhầm với vật giống phone bằng điểm âm và temporal ổn định thật sự
    """

    def __init__(self):
        self.confirm_frames_on = 4
        self.confirm_frames_off = 4

        self.use_counter = 0
        self.clear_counter = 0
        self.phone_using_state = False

        self.track_history = deque(maxlen=14)
        self.last_best_box = None

        self.wrist_phone_dist_ratio = 0.58
        self.elbow_phone_dist_ratio = 0.52
        self.face_phone_dist_ratio = 0.42
        self.thigh_phone_dist_ratio = 0.40

        self.use_score_threshold = 5  # lowered from 7 to catch more phone scenarios


    def _phone_in_driver_zone(self, phone_box, person_box):
        phone_box = safe_box(phone_box)
        person_box = safe_box(person_box)
        if phone_box is None or person_box is None:
            return False

        pc = box_center(phone_box)
        if pc is None:
            return False

        px1, py1, px2, py2 = person_box
        pw = px2 - px1
        ph = py2 - py1

        zx1 = px1 - pw * 0.18
        zx2 = px2 + pw * 0.18
        zy1 = py1 - ph * 0.05
        zy2 = py1 + ph * 1.18

        return zx1 <= pc[0] <= zx2 and zy1 <= pc[1] <= zy2

    def _phone_near_wrist(self, phone_box, wrist_points, person_box):
        if phone_box is None or person_box is None or not wrist_points:
            return False
        phone_c = box_center(phone_box)
        px1, py1, px2, py2 = map(float, person_box)
        pw = px2 - px1
        limit = pw * self.wrist_phone_dist_ratio
        return any(wp is not None and point_distance(phone_c, wp) <= limit for wp in wrist_points)

    def _phone_near_elbow(self, phone_box, elbow_points, person_box):
        if phone_box is None or person_box is None or not elbow_points:
            return False
        phone_c = box_center(phone_box)
        px1, py1, px2, py2 = map(float, person_box)
        pw = px2 - px1
        limit = pw * self.elbow_phone_dist_ratio
        return any(ep is not None and point_distance(phone_c, ep) <= limit for ep in elbow_points)

    def _phone_near_face(self, phone_box, face_center, person_box):
        if phone_box is None or face_center is None or person_box is None:
            return False
        phone_c = box_center(phone_box)
        px1, py1, px2, py2 = map(float, person_box)
        pw = px2 - px1
        limit = pw * self.face_phone_dist_ratio
        return point_distance(phone_c, face_center) <= limit

    def _phone_near_thigh(self, phone_box, thigh_points, person_box):
        if phone_box is None or person_box is None or not thigh_points:
            return False
        phone_c = box_center(phone_box)
        px1, py1, px2, py2 = map(float, person_box)
        pw = px2 - px1
        limit = pw * self.thigh_phone_dist_ratio
        return any(tp is not None and point_distance(phone_c, tp) <= limit for tp in thigh_points)

    def _is_rectangular_phone_like(self, phone_box):
        w, h = box_size(phone_box)
        if w <= 0 or h <= 0:
            return False
        ratio = w / float(h)
        area = w * h
        if area < 180:
            return False
        return 0.28 <= ratio <= 3.6

    def _phone_is_horizontalish(self, phone_box):
        w, h = box_size(phone_box)
        if w <= 0 or h <= 0:
            return False
        return (w / float(h)) >= 1.20

    def _update_track_and_get_temporal_features(self, phone_box, person_box):
        box = safe_box(phone_box)
        if box is None:
            self.track_history.clear()
            self.last_best_box = None
            return {
                "temporal_ok": False,
                "center_stable": False,
                "size_stable": False,
                "aspect_stable": False,
                "track_len": 0,
            }

        c = box_center(box)
        w, h = box_size(box)
        aspect = w / float(max(1.0, h))
        area = w * h

        self.track_history.append({
            "center": c,
            "w": w,
            "h": h,
            "aspect": aspect,
            "area": area,
        })
        self.last_best_box = box

        if len(self.track_history) < 3:
            return {
                "temporal_ok": False,
                "center_stable": False,
                "size_stable": False,
                "aspect_stable": False,
                "track_len": len(self.track_history),
            }

        xs = [it["center"][0] for it in self.track_history]
        ys = [it["center"][1] for it in self.track_history]
        ws = [it["w"] for it in self.track_history]
        hs = [it["h"] for it in self.track_history]
        aspects = [it["aspect"] for it in self.track_history]
        areas = [it["area"] for it in self.track_history]

        if person_box is not None:
            px1, py1, px2, py2 = map(float, person_box)
            pw = max(1.0, px2 - px1)
            ph = max(1.0, py2 - py1)
        else:
            pw = 220.0
            ph = 320.0

        center_stable = (max(xs) - min(xs)) <= pw * 0.38 and (max(ys) - min(ys)) <= ph * 0.34
        size_stable = (max(ws) - min(ws)) <= max(16.0, pw * 0.22) and (max(hs) - min(hs)) <= max(16.0, ph * 0.18)

        asp_max = max(aspects)
        asp_min = min(aspects)
        aspect_stable = asp_max <= max(1.18, asp_min * 1.75)

        area_max = max(areas)
        area_min = min(areas)
        area_stable = area_max <= max(220.0, area_min * 2.2)

        temporal_ok = center_stable and aspect_stable and area_stable

        return {
            "temporal_ok": bool(temporal_ok),
            "center_stable": bool(center_stable),
            "size_stable": bool(size_stable and area_stable),
            "aspect_stable": bool(aspect_stable),
            "track_len": len(self.track_history),
        }

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
        temporal_feats,
    ):
        score = 0

        in_driver_zone = self._phone_in_driver_zone(phone_box, person_box)
        near_wrist = self._phone_near_wrist(phone_box, wrist_points, person_box)
        near_elbow = self._phone_near_elbow(phone_box, elbow_points, person_box)
        near_face = self._phone_near_face(phone_box, face_center, person_box)
        near_thigh = self._phone_near_thigh(phone_box, thigh_points, person_box)

        temporal_ok = temporal_feats.get("temporal_ok", False)
        center_stable = temporal_feats.get("center_stable", False)
        size_stable = temporal_feats.get("size_stable", False)
        aspect_stable = temporal_feats.get("aspect_stable", False)

        horizontalish = self._phone_is_horizontalish(phone_box)
        rectangular_ok = self._is_rectangular_phone_like(phone_box)

        in_lap_zone = phone_ctx.get("in_lap_zone", False)
        in_mid_leg_zone = phone_ctx.get("in_mid_leg_zone", False)
        in_left_leg_zone = phone_ctx.get("in_left_leg_zone", False)
        in_right_leg_zone = phone_ctx.get("in_right_leg_zone", False)

        lap_overlap = float(phone_ctx.get("lap_overlap", 0.0))
        mid_leg_overlap = float(phone_ctx.get("mid_leg_overlap", 0.0))
        left_leg_overlap = float(phone_ctx.get("left_leg_overlap", 0.0))
        right_leg_overlap = float(phone_ctx.get("right_leg_overlap", 0.0))

        head_down = context_out.get("head_down", False)
        lap_attention = context_out.get("lap_attention", False)
        left_side_attention = context_out.get("left_side_attention", False)
        right_side_attention = context_out.get("right_side_attention", False)

        lap_attention_strength = float(context_out.get("lap_attention_strength", 0.0))
        left_side_attention_strength = float(context_out.get("left_side_attention_strength", 0.0))
        right_side_attention_strength = float(context_out.get("right_side_attention_strength", 0.0))

        if in_driver_zone:
            score += 2
        else:
            score -= 4

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
            score += 3
        if in_left_leg_zone or in_right_leg_zone:
            score += 2

        if lap_overlap >= 0.35:
            score += 2
        if mid_leg_overlap >= 0.35:
            score += 3
        if left_leg_overlap >= 0.35 or right_leg_overlap >= 0.35:
            score += 2

        if temporal_ok:
            score += 2
        if center_stable:
            score += 1
        if size_stable:
            score += 1
        if aspect_stable:
            score += 1

        if head_down:
            score += 1

        if lap_attention:
            score += 1
        if left_side_attention or right_side_attention:
            score += 1

        score += 2 * min(1.0, lap_attention_strength)
        score += 1 * min(1.0, left_side_attention_strength + right_side_attention_strength)

        if near_thigh and temporal_ok:
            score += 2

        if (in_lap_zone or in_mid_leg_zone) and head_down:
            score += 3

        if (in_lap_zone or in_mid_leg_zone) and (near_thigh or near_wrist or near_elbow):
            score += 3

        if (in_lap_zone or in_mid_leg_zone) and lap_attention and temporal_ok:
            score += 3

        if horizontalish and (in_lap_zone or in_mid_leg_zone or in_left_leg_zone or in_right_leg_zone):
            score += 2

        if horizontalish and head_down and temporal_ok:
            score += 2

        if (in_left_leg_zone and left_side_attention) or (in_right_leg_zone and right_side_attention):
            score += 3

        if (in_mid_leg_zone or in_left_leg_zone or in_right_leg_zone) and head_down and temporal_ok:
            score += 3

        if rectangular_ok:
            score += 1
        else:
            score -= 2

        if not near_wrist and not near_elbow and not near_thigh and not near_face:
            score -= 2

        if not (in_lap_zone or in_mid_leg_zone or in_left_leg_zone or in_right_leg_zone) and not near_face:
            score -= 2

        return int(round(score))

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
    ) -> PhoneUsageOut:
        context_out = context_out or {}
        phone_contexts = context_out.get("phone_contexts", [])

        wrist_points = [left_wrist, right_wrist]
        elbow_points = [left_elbow, right_elbow]
        thigh_points = [left_thigh_center, right_thigh_center]

        prelim_best_box = None
        prelim_best_ctx = None
        prelim_rank = -1e9

        for ctx in phone_contexts:
            pb = ctx.get("box")
            rank = 0.0
            if ctx.get("in_mid_leg_zone"):
                rank += 3.0
            if ctx.get("in_lap_zone"):
                rank += 2.0
            if ctx.get("in_left_leg_zone") or ctx.get("in_right_leg_zone"):
                rank += 2.0
            rank += 4.0 * float(ctx.get("mid_leg_overlap", 0.0))
            rank += 3.0 * float(ctx.get("lap_overlap", 0.0))
            rank += 2.5 * max(float(ctx.get("left_leg_overlap", 0.0)), float(ctx.get("right_leg_overlap", 0.0)))
            if rank > prelim_rank:
                prelim_rank = rank
                prelim_best_box = pb
                prelim_best_ctx = ctx

        if prelim_best_box is not None:
            temporal_feats = self._update_track_and_get_temporal_features(prelim_best_box, person_box)
        else:
            self.track_history.clear()
            self.last_best_box = None
            temporal_feats = {
                "temporal_ok": False,
                "center_stable": False,
                "size_stable": False,
                "aspect_stable": False,
                "track_len": 0,
            }

        best_phone_box = None
        best_score = 0.0
        best_source = None

        for ctx in phone_contexts:
            pb = ctx.get("box")
            feats = temporal_feats if pb == prelim_best_box else {
                "temporal_ok": False,
                "center_stable": False,
                "size_stable": False,
                "aspect_stable": False,
                "track_len": 0,
            }

            score = self._score_phone(
                pb,
                person_box,
                wrist_points,
                elbow_points,
                thigh_points,
                face_center,
                ctx,
                context_out,
                feats,
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
            self.track_history.clear()
            self.last_best_box = None

        return {
            "phone_using": self.phone_using_state,
            "best_phone_box": best_phone_box,
            "score": int(best_score),
            "source": best_source,
            "temporal_track_len": temporal_feats.get("track_len", 0),
        }

    def reset(self):
        self.use_counter = 0
        self.clear_counter = 0
        self.phone_using_state = False
        self.track_history.clear()
        self.last_best_box = None
