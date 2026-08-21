# -*- coding: utf-8 -*-
"""Phat hien vach dung ngang (Smart City).

Quan trong: DB §4.8 phan biet hai loi khac han nhau
  - "Khong dung truoc vach dung khi den do"  -> -10 diem
  - "Vuot den do"                            -> HUY LUOT
Nen xe phai biet vach dung o dau, khong chi biet den mau gi. Dung som van hon
dung tre: dung thua chi ton vai giay (0.1 diem/giay o Smart City).
"""

import cv2
import numpy as np


class StoplineResult(object):
    def __init__(self, found, distance, row=None):
        # distance: 0.0 = ngay truoc mui xe, 1.0 = xa nhat trong ROI
        self.found = found
        self.distance = distance
        self.row = row


class StoplineDetector(object):
    def __init__(self, cfg):
        self.pw = int(cfg.get('pipeline.proc_width', 320))
        self.ph = int(cfg.get('pipeline.proc_height', 240))
        self.roi_top = float(cfg.get('stopline.roi_top', 0.72))
        self.min_row_fill = float(cfg.get('stopline.min_row_fill', 0.45))
        self.min_rows = int(cfg.get('stopline.min_rows', 3))

    def process(self, frame_bgr):
        # Runner da dua frame ve proc_width/proc_height. Khong resize/copy lai
        # 30 lan/giay neu kich thuoc da dung; van giu fallback cho API doc lap.
        if frame_bgr.shape[1] == self.pw and frame_bgr.shape[0] == self.ph:
            small = frame_bgr
        else:
            small = cv2.resize(frame_bgr, (self.pw, self.ph))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        y0 = int(self.ph * self.roi_top)
        roi = binary[y0:, :]
        if roi.size == 0:
            return StoplineResult(False, 1.0)

        # Ti le pixel sang tren tung hang; vach ngang => hang gan nhu day
        fill = roi.sum(axis=1) / (255.0 * self.pw)
        hot = fill >= self.min_row_fill

        # Tim day hang lien tiep dai nhat
        best_len, best_end = 0, -1
        run_len = 0
        for i, is_hot in enumerate(hot):
            if is_hot:
                run_len += 1
                if run_len > best_len:
                    best_len, best_end = run_len, i
            else:
                run_len = 0

        if best_len < self.min_rows:
            return StoplineResult(False, 1.0)

        # Hang cuoi cua vach (gan xe nhat trong dai)
        row_in_roi = best_end
        row_abs = y0 + row_in_roi
        # Cang gan day anh => cang gan xe => distance cang nho
        distance = float(np.clip((self.ph - row_abs) / float(self.ph - y0), 0.0, 1.0))
        return StoplineResult(True, distance, row_abs)
