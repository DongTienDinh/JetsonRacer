# -*- coding: utf-8 -*-
"""Nhan dien bien bao / den giao thong + latch quyet dinh.

Hai backend:
  - StubDetector : khong can model, dung de dev offline va chay test.
  - YoloDetector : YOLOv8n export TensorRT FP16 (chay tren xe).

Bat doi xung co chu dich giua den do va den xanh:
  conf_red  thap  -> tha bao do thua (chi ton vai giay dung)
  conf_green cao  -> bao nham xanh = vuot den do = HUY LUOT (DB §4.8)
"""

import time

import numpy as np


class Detection(object):
    def __init__(self, label, confidence, bbox, area_ratio):
        self.label = label
        self.confidence = confidence
        self.bbox = bbox                  # (x1, y1, x2, y2) trong anh goc
        self.area_ratio = area_ratio      # dien tich bbox / dien tich anh
        self.t_first_seen = None          # dat boi SignTracker, phuc vu latency_ms


class BaseDetector(object):
    def detect(self, frame_bgr):
        """Tra ve list[Detection]."""
        raise NotImplementedError


class StubDetector(BaseDetector):
    """Khong nhan dien gi. Cho phep chay toan bo pipeline khi chua co model.

    Co the nap kich ban dung san (danh sach (frame_index, label, conf, area))
    de kiem thu FSM mot cach xac dinh - dung trong tests/.
    """

    def __init__(self, script=None):
        self.script = script or []
        self._i = 0

    def detect(self, frame_bgr):
        out = []
        for item in self.script:
            if item[0] == self._i:
                label, conf, area = item[1], item[2], item[3]
                out.append(Detection(label, conf, (0, 0, 10, 10), area))
        self._i += 1
        return out


class YoloDetector(BaseDetector):
    """YOLOv8 (ultralytics) hoac engine TensorRT da export.

    Import duoc lam lazy: laptop khong cai ultralytics van chay duoc backend stub.
    """

    def __init__(self, model_path, input_size=320, conf_threshold=0.5, classes=None):
        from ultralytics import YOLO  # noqa: import trong ham la co y

        self.model = YOLO(model_path)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.classes = classes or []

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        results = self.model.predict(
            frame_bgr, imgsz=self.input_size, conf=self.conf_threshold, verbose=False,
        )
        out = []
        area_img = float(h * w)
        for res in results:
            boxes = getattr(res, 'boxes', None)
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = [float(v) for v in box.xyxy[0]]
                label = self.classes[cls_id] if cls_id < len(self.classes) else str(cls_id)
                bw = max(0.0, xyxy[2] - xyxy[0])
                bh = max(0.0, xyxy[3] - xyxy[1])
                out.append(Detection(label, conf, tuple(xyxy), (bw * bh) / area_img))
        return out


def build_detector(cfg, script=None):
    backend = cfg.get('signs.backend', 'stub')
    if backend == 'yolo':
        return YoloDetector(
            cfg.get('signs.model_path'),
            input_size=int(cfg.get('signs.input_size', 320)),
            conf_threshold=float(cfg.get('signs.conf_threshold', 0.5)),
            classes=cfg.get('signs.classes', []),
        )
    return StubDetector(script=script)


TRAFFIC_LIGHTS = ('red_light', 'green_light')


class SignTracker(object):
    """Gom detection qua nhieu frame -> nhan on dinh + moc thoi gian cho latency.

    Ly do phai vote: mot frame nhieu co the doc `no_left` thanh `turn_left`.
    Sai 1 nga tu = di sai lo trinh = KET THUC LUOT (DB §4.8).
    """

    def __init__(self, cfg):
        self.k = int(cfg.get('signs.vote_k', 3))
        self.n = int(cfg.get('signs.vote_n', 5))
        self.conf_default = float(cfg.get('signs.conf_threshold', 0.55))
        self.conf_green = float(cfg.get('signs.conf_green', 0.80))
        self.conf_red = float(cfg.get('signs.conf_red', 0.35))
        self.lock_area = float(cfg.get('signs.lock_area_ratio', 0.02))

        self._history = []                # list cac list nhan cua tung frame
        self._first_seen = {}             # label -> timestamp lan dau thay
        self.last_stable = None           # (label, confidence)
        self.last_first_seen = None

    def _threshold_for(self, label):
        if label == 'green_light':
            return self.conf_green
        if label == 'red_light':
            return self.conf_red
        return self.conf_default

    def update(self, detections, now=None):
        """Tra ve (label, confidence, t_first_seen) neu co nhan on dinh, else None."""
        now = time.time() if now is None else now

        accepted = []
        for det in detections:
            if det.confidence >= self._threshold_for(det.label):
                accepted.append(det)
                if det.label not in self._first_seen:
                    self._first_seen[det.label] = now

        # Chi latch bien bao dieu huong khi da du gan (bbox du lon).
        # Den giao thong khong ap dieu kien nay: phai phan ung tu xa de kip dung.
        eligible = [
            d for d in accepted
            if d.label in TRAFFIC_LIGHTS or d.area_ratio >= self.lock_area
        ]

        self._history.append([d.label for d in eligible])
        if len(self._history) > self.n:
            self._history.pop(0)

        counts = {}
        for labels in self._history:
            for label in set(labels):
                counts[label] = counts.get(label, 0) + 1

        winner = None
        for label, count in counts.items():
            if count >= self.k:
                if winner is None or count > counts[winner]:
                    winner = label

        if winner is None:
            return None

        conf = max([d.confidence for d in eligible if d.label == winner] or [0.0])
        first_seen = self._first_seen.get(winner, now)
        self.last_stable = (winner, conf)
        self.last_first_seen = first_seen
        return winner, conf, first_seen

    def clear(self, label=None):
        """Xoa trang thai sau khi da thuc thi xong mot bien bao."""
        self._history = []
        if label is None:
            self._first_seen = {}
        else:
            self._first_seen.pop(label, None)
        self.last_stable = None
        self.last_first_seen = None
