# -*- coding: utf-8 -*-
"""Bam lane bang CV co dien: bird's-eye warp -> nguong thich nghi -> histogram peak.

Vi sao khong dung CNN o baseline: DB §7 noi ro dataset BTC KHONG giong sa ban thi.
CV co dien chinh duoc tai cho bang 2-3 tham so nguong trong 5 phut chuan bi;
CNN can thu data + train lai. CNN la CAI TIEN (thi nghiem E1), khong phai baseline.

Dau ra chinh: cross-track error `cte` chuan hoa trong [-1, 1].
    cte < 0 -> tam lane lech sang TRAI so voi xe -> can danh lai trai.
"""

import cv2
import numpy as np


class LaneResult(object):
    def __init__(self, found, cte, curvature, n_pixels, debug=None):
        self.found = found
        self.cte = cte
        self.curvature = curvature
        self.n_pixels = n_pixels
        self.debug = debug


class LaneDetector(object):
    def __init__(self, cfg):
        self.pw = int(cfg.get('pipeline.proc_width', 320))
        self.ph = int(cfg.get('pipeline.proc_height', 240))
        self.roi_top = float(cfg.get('lane.roi_top', 0.55))
        self.roi_bottom = float(cfg.get('lane.roi_bottom', 1.0))
        self.block = int(cfg.get('lane.threshold_block', 31))
        if self.block % 2 == 0:
            self.block += 1  # adaptiveThreshold yeu cau so le
        self.c = float(cfg.get('lane.threshold_c', -12))
        self.min_pixels = int(cfg.get('lane.min_pixels', 60))
        self.alpha = float(cfg.get('lane.smooth_alpha', 0.6))

        src = cfg.get('lane.warp_src')
        margin = float(cfg.get('lane.warp_dst_margin', 0.25))
        self._M = self._build_warp(src, margin)

        self._cte_ema = 0.0
        self._prev_cte = 0.0
        self._initialised = False

    def _build_warp(self, src_ratio, margin):
        w, h = float(self.pw), float(self.ph)
        src = np.float32([[p[0] * w, p[1] * h] for p in src_ratio])
        dst = np.float32([
            [margin * w, h],
            [(1.0 - margin) * w, h],
            [(1.0 - margin) * w, 0.0],
            [margin * w, 0.0],
        ])
        return cv2.getPerspectiveTransform(src, dst)

    def _binarise(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # Chuan hoa do sang: giam anh huong khi doi dieu kien anh sang (rui ro E5)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
            self.block, self.c,
        )
        kernel = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    def process(self, frame_bgr):
        small = cv2.resize(frame_bgr, (self.pw, self.ph))
        binary = self._binarise(small)

        y0 = int(self.ph * self.roi_top)
        y1 = int(self.ph * self.roi_bottom)
        mask = np.zeros_like(binary)
        mask[y0:y1, :] = binary[y0:y1, :]

        warped = cv2.warpPerspective(mask, self._M, (self.pw, self.ph))

        n_pixels = int(np.count_nonzero(warped))
        if n_pixels < self.min_pixels:
            # Mat lane: giu cte cu de xe khong giat banh lai dot ngot
            return LaneResult(False, self._cte_ema, 0.0, n_pixels, warped)

        # Chia doc thanh 3 dai: gan / trung / xa -> tam lane tung dai
        centers = []
        bands = 3
        band_h = self.ph // bands
        for b in range(bands):
            band = warped[self.ph - (b + 1) * band_h: self.ph - b * band_h, :]
            hist = band.sum(axis=0).astype(np.float32)
            total = hist.sum()
            if total <= 0:
                centers.append(None)
                continue
            xs = np.arange(self.pw, dtype=np.float32)
            centers.append(float((hist * xs).sum() / total))

        near = centers[0] if centers[0] is not None else self.pw / 2.0
        far = None
        for c in centers[1:]:
            if c is not None:
                far = c
                break

        # cte chuan hoa: 0 = giua anh, +-1 = mep anh
        raw_cte = (near - self.pw / 2.0) / (self.pw / 2.0)
        raw_cte = float(np.clip(raw_cte, -1.0, 1.0))

        # Do cong xap xi: lech giua dai xa va dai gan
        curvature = 0.0
        if far is not None:
            curvature = float(np.clip((far - near) / (self.pw / 2.0), -1.0, 1.0))

        if not self._initialised:
            self._cte_ema = raw_cte
            self._initialised = True
        else:
            self._cte_ema = self.alpha * raw_cte + (1.0 - self.alpha) * self._cte_ema

        self._prev_cte = self._cte_ema
        return LaneResult(True, self._cte_ema, curvature, n_pixels, warped)

    def debug_process(self, frame_bgr):
        """Nhu process() nhung tra ve ca cac buoc trung gian, de tune nguong.

        Tach rieng khoi process() co chu dich: process() la duong nong chay 30 Hz
        tren xe, khong duoc ganh them chi phi dung debug.
        """
        small = cv2.resize(frame_bgr, (self.pw, self.ph))
        binary = self._binarise(small)

        y0 = int(self.ph * self.roi_top)
        y1 = int(self.ph * self.roi_bottom)
        masked = np.zeros_like(binary)
        masked[y0:y1, :] = binary[y0:y1, :]
        warped = cv2.warpPerspective(masked, self._M, (self.pw, self.ph))

        result = self.process(frame_bgr)
        return {
            'small': small,
            'binary': binary,
            'masked': masked,
            'warped': warped,
            'roi_y': (y0, y1),
            'result': result,
        }

    def reset(self):
        self._cte_ema = 0.0
        self._prev_cte = 0.0
        self._initialised = False
