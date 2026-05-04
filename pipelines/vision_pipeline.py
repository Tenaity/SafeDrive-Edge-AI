import os
import time
import cv2
import numpy as np
import requests

from utils.box_utils import safe_box, box_area, intersection_area, point_distance
from utils.types import VisionOut


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return int(default)


class VisionPipeline:
    """
    VisionPipeline:
    - Detect toàn khung
    - Chỉ detect thêm ROI quanh tay khi toàn khung chưa thấy phone
    - ROI quanh tay nhỏ hơn để phone trong tay rõ hơn
    - Đọc tham số từ file .env.phone để dễ chỉnh mà không phải sửa code
    """

    def __init__(self):
        self.yolo_url = os.getenv("YOLO_URL", "http://127.0.0.1:8000/detect")
        print(f"[YOLO URL] {self.yolo_url}")

        self.TIMEOUT_SEC = env_float("VISION_TIMEOUT_SEC", 2.0)
        self.last_call = 0.0
        self.CALL_EVERY_SEC = env_float("VISION_CALL_EVERY_SEC", 0.22)

        self.last_out = {
            "phone": False,
            "dets": [],
            "persons": [],
            "phones": []
        }

        self.SEND_LONG_SIDE = env_int("VISION_SEND_LONG_SIDE", 1280)
        self.JPEG_QUALITY = env_int("VISION_JPEG_QUALITY", 86)
        self.CLIENT_PHONE_CONF_MIN = env_float("VISION_CLIENT_PHONE_CONF_MIN", 0.50)

        # Hand ROI
        self.MAX_HANDS_FOR_ROI = env_int("VISION_MAX_HANDS_FOR_ROI", 1)
        self.HAND_ROI_HALF_SIZE = env_int("VISION_HAND_ROI_HALF_SIZE", 140)
        self.HAND_ROI_UPSCALE = env_float("VISION_HAND_ROI_UPSCALE", 1.35)
        self.HAND_ROI_CONF_BONUS = env_float("VISION_HAND_ROI_CONF_BONUS", 0.03)

        self.consecutive_failures = 0
        self.MAX_FAILURES_BEFORE_CLEAR = 3

        self.session = requests.Session()

        print(
            "[VISION CFG]",
            "timeout=", self.TIMEOUT_SEC,
            "call_every=", self.CALL_EVERY_SEC,
            "send_long_side=", self.SEND_LONG_SIDE,
            "jpeg_quality=", self.JPEG_QUALITY,
            "phone_conf_min=", self.CLIENT_PHONE_CONF_MIN,
            "max_hands_roi=", self.MAX_HANDS_FOR_ROI,
            "hand_roi_half=", self.HAND_ROI_HALF_SIZE,
            "hand_roi_upscale=", self.HAND_ROI_UPSCALE,
            "hand_roi_conf_bonus=", self.HAND_ROI_CONF_BONUS,
        )

    def _resize_keep_ratio(self, frame, long_side):
        h, w = frame.shape[:2]
        if max(h, w) <= long_side:
            return frame, 1.0
        scale = long_side / float(max(h, w))
        nw = int(w * scale)
        nh = int(h * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return resized, scale

    def _clear_output(self):
        self.last_out = {
            "phone": False,
            "dets": [],
            "persons": [],
            "phones": []
        }


    def _normalize_output(self, data):
        if not isinstance(data, dict):
            return None

        dets = data.get("dets", [])
        persons = data.get("persons", [])
        phones = data.get("phones", [])

        if not isinstance(dets, list):
            dets = []
        if not isinstance(persons, list):
            persons = []
        if not isinstance(phones, list):
            phones = []

        return {
            "phone": bool(len(phones) > 0),
            "dets": dets,
            "persons": persons,
            "phones": phones,
        }

    def _request_detect(self, frame):
        ok, jpg = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.JPEG_QUALITY)]
        )
        if not ok:
            return None

        r = self.session.post(
            self.yolo_url,
            files={"image": ("frame.jpg", jpg.tobytes(), "image/jpeg")},
            timeout=self.TIMEOUT_SEC,
        )
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict) or not data.get("ok"):
            return None
        return self._normalize_output(data)

    def _dedup_phones(self, phones, frame_shape):
        if not phones:
            return []

        fh, fw = frame_shape[:2]
        area_ref = float(fw * fh)

        phones_sorted = sorted(
            phones,
            key=lambda p: (float(p.get("conf", 0.0)), box_area(p.get("xyxy"))),
            reverse=True
        )

        kept = []
        for ph in phones_sorted:
            box = ph.get("xyxy")
            if box is None:
                continue

            dup = False
            for kp in kept:
                kb = kp.get("xyxy")
                if kb is None:
                    continue
                inter = intersection_area(box, kb)
                min_area = min(box_area(box), box_area(kb))
                if min_area > 1e-6 and inter / min_area >= 0.65:
                    dup = True
                    break

                cx1 = (float(box[0]) + float(box[2])) / 2.0
                cy1 = (float(box[1]) + float(box[3])) / 2.0
                cx2 = (float(kb[0]) + float(kb[2])) / 2.0
                cy2 = (float(kb[1]) + float(kb[3])) / 2.0
                dist2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
                if dist2 <= max(36.0, 0.0004 * area_ref):
                    a1 = box_area(box)
                    a2 = box_area(kb)
                    if max(a1, a2) > 1e-6 and min(a1, a2) / max(a1, a2) >= 0.55:
                        dup = True
                        break

            if not dup:
                kept.append(ph)

        return kept

    def _filter_phones(self, phones):
        if not phones:
            return []

        return [
            p for p in phones
            if float(p.get("conf", 0.0)) >= self.CLIENT_PHONE_CONF_MIN and p.get("xyxy") is not None
        ]

    def _scale_hand_centers(self, hand_centers, scale):
        out = []
        for hc in hand_centers or []:
            if hc is None or len(hc) != 2:
                continue
            try:
                x = int(float(hc[0]) * scale)
                y = int(float(hc[1]) * scale)
                out.append((x, y))
            except Exception:
                continue
        return out

    def _pick_best_hand_centers(self, hand_centers):
        if not hand_centers:
            return []
        return hand_centers[:self.MAX_HANDS_FOR_ROI]

    def _build_hand_crops(self, frame, hand_centers):
        h, w = frame.shape[:2]
        crops = []

        for hc in hand_centers:
            cx, cy = hc

            half = self.HAND_ROI_HALF_SIZE
            x1 = max(0, cx - half)
            y1 = max(0, cy - half)
            x2 = min(w, cx + half)
            y2 = min(h, cy + half)

            crop = frame[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                continue

            up = cv2.resize(
                crop,
                None,
                fx=self.HAND_ROI_UPSCALE,
                fy=self.HAND_ROI_UPSCALE,
                interpolation=cv2.INTER_CUBIC
            )
            crops.append((up, (x1, y1, x2, y2)))

        return crops

    def _map_box_from_crop(self, box, crop_info, scale_x, scale_y):
        if box is None or crop_info is None:
            return box

        try:
            x1, y1, x2, y2 = map(float, box)
        except Exception:
            return box

        off_x1, off_y1, _, _ = crop_info
        return [
            x1 / scale_x + off_x1,
            y1 / scale_y + off_y1,
            x2 / scale_x + off_x1,
            y2 / scale_y + off_y1,
        ]

    def run(self, frame: np.ndarray, hand_centers: list | None = None) -> VisionOut:
        if frame is None or not hasattr(frame, "shape") or frame.size == 0:
            return self.last_out

        now = time.time()
        if now - self.last_call < self.CALL_EVERY_SEC:
            return self.last_out
        self.last_call = now

        send_frame, scale = self._resize_keep_ratio(frame, self.SEND_LONG_SIDE)
        scaled_hand_centers = self._scale_hand_centers(hand_centers or [], scale)

        try:
            main_out = self._request_detect(send_frame)
            if main_out is None:
                raise RuntimeError("invalid detect response")

            persons = list(main_out.get("persons", []))
            phones = list(main_out.get("phones", []))

            if len(phones) == 0 and scaled_hand_centers:
                picked_hands = self._pick_best_hand_centers(scaled_hand_centers)
                hand_crops = self._build_hand_crops(send_frame, picked_hands)

                for crop_img, crop_box in hand_crops:
                    roi_out = self._request_detect(crop_img)
                    if roi_out is None:
                        continue

                    scale_x = float(crop_img.shape[1]) / float(crop_box[2] - crop_box[0])
                    scale_y = float(crop_img.shape[0]) / float(crop_box[3] - crop_box[1])

                    for ph in roi_out.get("phones", []):
                        mapped = dict(ph)
                        mapped["xyxy"] = self._map_box_from_crop(
                            ph.get("xyxy"),
                            crop_box,
                            scale_x,
                            scale_y
                        )
                        mapped["conf"] = min(
                            0.99,
                            float(mapped.get("conf", 0.0)) + self.HAND_ROI_CONF_BONUS
                        )
                        phones.append(mapped)

            phones = self._dedup_phones(phones, send_frame.shape)
            phones = self._filter_phones(phones)

            dets = []
            for p in persons:
                dets.append({
                    "cls": int(p.get("cls", 0)),
                    "conf": float(p.get("conf", 1.0)),
                    "xyxy": p.get("xyxy"),
                })
            for ph in phones:
                dets.append({
                    "cls": int(ph.get("cls", 67)),
                    "conf": float(ph.get("conf", 0.0)),
                    "xyxy": ph.get("xyxy"),
                })

            self.consecutive_failures = 0
            self.last_out = {
                "phone": bool(len(phones) > 0),
                "dets": dets,
                "persons": persons,
                "phones": phones
            }
            return self.last_out

        except Exception as e:
            print(f"Vision API error: {e}")
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.MAX_FAILURES_BEFORE_CLEAR:
                self._clear_output()
            return self.last_out

    def close(self):
        try:
            self.session.close()
        except Exception:
            pass