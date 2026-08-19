# -*- coding: utf-8 -*-
"""Bam VACH DUT O GIUA sa ban bang CV co dien.

Vi sao khong dung CNN o baseline: DB §7 noi ro dataset BTC KHONG giong sa ban thi.
CV co dien chinh duoc tai cho bang 2-3 tham so nguong trong 5 phut chuan bi;
CNN can thu data + train lai. CNN la CAI TIEN (thi nghiem E1), khong phai baseline.

MUC TIEU BAM: vach dut o CHINH GIUA lane, khong phai vien lane.

MAU VACH LA THAM SO, khong hardcode:
  - `white`: DB §3.2 mo ta sa ban thi - "lane mau toi, line trang dut khuc o giua".
  - `red`  : sa ban tap hien tai - vach dut do tren nen san sang.
Cung mot thuat toan, chi doi mask mau. Doi sa ban thi doi `lane.line_color`,
khong sua code.

VI SAO KHONG DUNG NGUONG XAM NHU BAN CU: tren sa ban tap, adaptiveThreshold tren
anh xam khoa vao VIEN TRANG chu khong phai vach do o giua - vach do va san xam co
do sang gan bang nhau nen bien mat sau BGR2GRAY. Do duoc tren raw_camera.avi: ban
cu cho cte_rms 0.211, lai bao hoa +-0.60 o 31.5% frame va doi dau 4.8 lan/giay.

CACH BAM VACH DUT (van de rieng cua vach dut: no BIEN MAT theo chu ky):
  1. Mask mau -> ROI -> bird's-eye warp.
     Warp lam dai phan xa ra nen cac net dut o xa to len: so band hop le tang tu
     4.0/10 len 7.5/10 (do tren 297 frame raw_camera.avi).
  2. Chia doc thanh N band. Trong moi band, tim cac CUM cot lien tiep co pixel,
     roi chon cum GAN NHAT voi vi tri du doan - khong lay trong tam cua ca band.
     Day la khac biet quan trong: lay trong tam ca band se gop ca vien lane vao
     va cho ra mot con so khong lien he gi voi tam vach.
  3. Band nao trong (dang o giua hai net dut) thi bo qua, khong noi suy bua.
  4. Fit da thuc bac 2 qua cac diem con lai -> vach dut roi rac thanh mot duong
     lien tuc. Day la ly do fit thay vi dung diem gan nhat: net dut co the vua
     het ngay truoc mui xe.
  5. Tu da thuc suy ra: `cte` (lech tai muc xe), `cte_lookahead` (lech tai diem
     ngam xa) va `curvature` (do cong) -> dung cho ga theo cua.
"""

import cv2
import numpy as np


class LaneResult(object):
    def __init__(self, found, cte, curvature, n_pixels, debug=None,
                 cte_lookahead=None, n_bands=0, fit=None):
        self.found = found
        self.cte = cte
        self.curvature = curvature
        self.n_pixels = n_pixels
        self.debug = debug
        # Lech tai diem ngam xa. Dung de danh lai SOM khi vao cua thay vi doi
        # den luc xe da lech roi moi sua.
        self.cte_lookahead = cte if cte_lookahead is None else cte_lookahead
        self.n_bands = n_bands          # so band tim thay vach - do tin cay
        self.fit = fit                  # he so da thuc, de ve debug


# Nguong mau mac dinh cho tung loai vach. `V` la kenh sang trong HSV.
# CHOT LAI BANG DATA THAT - day chi la diem xuat phat da kiem chung tren
# raw_camera.avi (sau khi da sua lens shading).
COLOR_PRESETS = {
    # Vach do: hue quanh 0/180. S phai du cao de loai san xam.
    # S > 80 bat duoc vach o 99/99 frame mau; S > 120 tut con 91/99.
    'red': {
        'hsv_low_1': [0, 80, 70], 'hsv_high_1': [10, 255, 255],
        'hsv_low_2': [170, 80, 70], 'hsv_high_2': [180, 255, 255],
        # 25 do tren raw_camera.avi: 15 -> mat vach 34.6%, 25 -> 0.40%.
        # San bong tao mot cum dom nhieu ngay trong khoang 15-25 px.
        'min_blob_area': 25,
    },
    # Vach trang tren lane toi (sa ban thi theo DB §3.2): S THAP, V CAO.
    'white': {
        'hsv_low_1': [0, 0, 170], 'hsv_high_1': [180, 60, 255],
        'hsv_low_2': None, 'hsv_high_2': None,
        # CHUA KIEM CHUNG TREN SA BAN THAT. 12 chi moi kiem tra tren nguon anh
        # tong hop (net dut rong 2 px) - do rong do la tu dat, khong phai do tu
        # sa ban thi. PHAI do lai bang video that o buoc T3.
        'min_blob_area': 12,
    },
}


class LaneDetector(object):
    def __init__(self, cfg):
        self.pw = int(cfg.get('pipeline.proc_width', 320))
        self.ph = int(cfg.get('pipeline.proc_height', 240))
        self.roi_top = float(cfg.get('lane.roi_top', 0.55))
        self.roi_bottom = float(cfg.get('lane.roi_bottom', 1.0))
        self.alpha = float(cfg.get('lane.smooth_alpha', 0.6))

        self.mode = str(cfg.get('lane.mode', 'color_center'))
        self.line_color = str(cfg.get('lane.line_color', 'red'))

        preset = COLOR_PRESETS.get(self.line_color, COLOR_PRESETS['red'])

        # `hsv_s_min` / `hsv_v_min` la hai num CHINH THEO MAU-DOC-LAP: chung chi
        # nang san S/V cua preset dang dung. Giao dien tune chi ghi hai so nay,
        # KHONG ghi ca dai HSV. Ly do: neu ghi thang `hsv_low_1` cho vach do roi
        # sau nay doi `line_color: white`, dai mau do cu se de len preset trang
        # ma khong bao gi - loi im lang, rat kho tim.
        s_min = cfg.get('lane.hsv_s_min')
        v_min = cfg.get('lane.hsv_v_min')

        def _floor(low):
            if low is None:
                return None
            low = list(low)
            if s_min is not None:
                low[1] = int(s_min)
            if v_min is not None:
                low[2] = int(v_min)
            return low

        self.hsv_low_1 = np.array(
            _floor(cfg.get('lane.hsv_low_1', preset['hsv_low_1'])), np.uint8)
        self.hsv_high_1 = np.array(
            cfg.get('lane.hsv_high_1', preset['hsv_high_1']), np.uint8)
        low2 = _floor(cfg.get('lane.hsv_low_2', preset['hsv_low_2']))
        high2 = cfg.get('lane.hsv_high_2', preset['hsv_high_2'])
        self.hsv_low_2 = None if low2 is None else np.array(low2, np.uint8)
        self.hsv_high_2 = None if high2 is None else np.array(high2, np.uint8)

        self.n_bands = int(cfg.get('lane.n_bands', 10))
        self.band_min_pixels = int(cfg.get('lane.band_min_pixels', 25))
        self.min_bands = int(cfg.get('lane.min_bands', 2))
        # Cum cot rong hon nguong nay khong phai vach dut (thuong la vien lane
        # hoac vung loa) -> loai truoc khi chon.
        self.max_run_frac = float(cfg.get('lane.max_run_frac', 0.45))
        # Cum nam xa vi tri du doan hon nguong nay -> khong phai vach dang bam.
        self.max_jump_frac = float(cfg.get('lane.max_jump_frac', 0.35))
        self.lookahead = float(cfg.get('lane.lookahead', 0.6))
        # KHU NHIEU THEO DIEN TICH, KHONG DUNG MORPH_OPEN.
        # Do duoc ca hai chieu tren du lieu that:
        #   - Bat MORPH_OPEN 3x3 : vach do OK, nhung vach TRANG manh bi erode
        #     xoa sach -> synthetic (vach trang) mat vach 120/120 frame.
        #   - Tat MORPH_OPEN     : vach trang OK, nhung nhieu dom tren san bong
        #     lam vach do mat 34.4% frame.
        # Loai blob nho hon `min_blob_area` giai quyet ca hai: dom nhieu co dien
        # tich 1-10 px bi bo, con mot net dut rong 2 px cao 20 px (dien tich 40)
        # van con nguyen. Erode khong phan biet duoc hai truong hop do.
        # Nguong theo do RONG cua vach nen khac nhau giua hai sa ban -> lay
        # mac dinh tu preset mau, khong dung mot so chung.
        self.min_blob_area = int(
            cfg.get('lane.min_blob_area', preset['min_blob_area']))
        self.morph_close = int(cfg.get('lane.morph_close', 3))

        # --- tham so cua che do xam cu (giu lai de so sanh trong paper) -------
        self.block = int(cfg.get('lane.threshold_block', 31))
        if self.block % 2 == 0:
            self.block += 1
        self.c = float(cfg.get('lane.threshold_c', -12))
        self.min_pixels = int(cfg.get('lane.min_pixels', 60))

        src = cfg.get('lane.warp_src')
        margin = float(cfg.get('lane.warp_dst_margin', 0.25))
        self._M = self._build_warp(src, margin)

        self._cte_ema = 0.0
        self._look_ema = 0.0
        self._curv_ema = 0.0
        self._anchor = None          # vi tri vach o frame truoc (pixel)
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

    # --------------------------------------------------------------- masking
    def _mask_color(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_low_1, self.hsv_high_1)
        if self.hsv_low_2 is not None:
            mask = cv2.bitwise_or(
                mask, cv2.inRange(hsv, self.hsv_low_2, self.hsv_high_2))
        if self.min_blob_area > 0:
            mask = self._drop_small_blobs(mask)
        if self.morph_close > 0:
            # Close: noi lien mot net dut bi rach doi. An toan voi vach manh.
            k = np.ones((self.morph_close, self.morph_close), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    def _drop_small_blobs(self, mask):
        n, labels, stats, _c = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1:
            return mask
        keep = (stats[:, cv2.CC_STAT_AREA] >= self.min_blob_area)
        keep[0] = False                      # nhan 0 la nen
        # LUT mot buoc thay vi np.where (np.where quet mang them mot lan nua).
        lut = np.where(keep, np.uint8(255), np.uint8(0))
        return lut[labels]

    def _mask_gray(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
            self.block, self.c,
        )
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def _binarise(self, bgr):
        if self.mode == 'gray':
            return self._mask_gray(bgr)
        return self._mask_color(bgr)

    # ------------------------------------------------------- band extraction
    def _band_candidate(self, band, anchor):
        """Tam cum cot gan `anchor` nhat trong mot band. None neu khong co.

        Tim theo CUM chu khong theo trong tam ca band: tren sa ban thi, vien lane
        cung mau trang voi vach giua, lay trong tam se keo diem ngam ra vien.
        """
        cols = np.count_nonzero(band, axis=0)
        if cols.sum() < self.band_min_pixels:
            return None, 0

        hot = cols > 0
        # Bien cua cac cum cot lien tiep
        edges = np.diff(np.concatenate(([0], hot.view(np.int8), [0])))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        if len(starts) == 0:
            return None, 0

        max_run = self.max_run_frac * self.pw
        best_x, best_n, best_d = None, 0, None
        for s, e in zip(starts, ends):
            if (e - s) > max_run:
                continue                      # qua rong -> vien lane, khong phai vach
            weights = cols[s:e].astype(np.float32)
            total = weights.sum()
            if total < self.band_min_pixels:
                continue
            xs = np.arange(s, e, dtype=np.float32)
            cx = float((weights * xs).sum() / total)
            d = abs(cx - anchor)
            if best_d is None or d < best_d:
                best_x, best_n, best_d = cx, int(total), d

        if best_x is None:
            return None, 0
        if best_d is not None and best_d > self.max_jump_frac * self.pw:
            return None, 0                    # nhay qua xa -> khong phai vach dang bam
        return best_x, best_n

    def _track_bands(self, warped):
        """Tra ve (danh sach diem (t, x), tong so pixel). t = 0 gan xe, 1 xa nhat."""
        band_h = max(1, self.ph // self.n_bands)
        anchor = self._anchor if self._anchor is not None else self.pw / 2.0
        points, total = [], 0
        for b in range(self.n_bands):
            y1 = self.ph - b * band_h
            y0 = max(0, y1 - band_h)
            if y1 <= 0:
                break
            cx, n = self._band_candidate(warped[y0:y1, :], anchor)
            if cx is None:
                continue                      # khoang trong giua hai net dut
            total += n
            t = (b + 0.5) / float(self.n_bands)
            points.append((t, cx))
            anchor = cx                       # bam tiep tu cum vua tim duoc
        return points, total

    # ----------------------------------------------------------------- fit
    @staticmethod
    def _fit(points):
        """x = a*t^2 + b*t + c. Bac 1 khi chi co 2 diem."""
        ts = np.array([p[0] for p in points], np.float64)
        xs = np.array([p[1] for p in points], np.float64)
        deg = 2 if len(points) >= 3 else 1
        coeff = np.polyfit(ts, xs, deg)
        if deg == 1:
            coeff = np.array([0.0, coeff[0], coeff[1]])
        return coeff

    def process(self, frame_bgr):
        small = frame_bgr
        if small.shape[1] != self.pw or small.shape[0] != self.ph:
            small = cv2.resize(small, (self.pw, self.ph))

        # CHI xu ly vung ROI. Ban cu tao mask ca khung roi moi zero phan tren -
        # tra tien cvtColor / inRange / loc blob cho 55% so pixel bi vut di ngay
        # sau do. Cat truoc tiet kiem dung phan do, va loc blob tro nen dung hon:
        # blob vat qua mep ROI khong con duoc tinh dien tich tu phan ngoai ROI.
        y0 = int(self.ph * self.roi_top)
        y1 = int(self.ph * self.roi_bottom)
        mask = np.zeros((self.ph, self.pw), np.uint8)
        if y1 > y0:
            mask[y0:y1, :] = self._binarise(small[y0:y1, :])
        warped = cv2.warpPerspective(mask, self._M, (self.pw, self.ph))

        points, n_pixels = self._track_bands(warped)

        if len(points) < self.min_bands:
            # Mat vach: giu gia tri cu de xe khong giat banh lai dot ngot.
            # KHONG reset _anchor - vach dut se quay lai gan cho cu.
            return LaneResult(False, self._cte_ema, self._curv_ema, n_pixels,
                              warped, cte_lookahead=self._look_ema,
                              n_bands=len(points))

        coeff = self._fit(points)
        half = self.pw / 2.0

        def x_at(t):
            return coeff[0] * t * t + coeff[1] * t + coeff[2]

        # CHI DUOC DUNG DA THUC TRONG VUNG THUC SU NHIN THAY VACH.
        # Vach dut nen doi khi chi thay 2-3 net o gan; ngoai suy bac 2 ra toi
        # t = 1.0 tu do cho ra so vo nghia. Do duoc tren raw_camera.avi: ngoai
        # suy lam do cong bao hoa +-1.0 o 28.6% frame (ga tut ve v_min suot
        # luot chay), con gioi han trong vung quan sat thi chi 1.7%.
        t_max = max(p[0] for p in points)
        t_look = min(self.lookahead, t_max)

        raw_cte = float(np.clip((x_at(0.0) - half) / half, -1.0, 1.0))
        raw_look = float(np.clip((x_at(t_look) - half) / half, -1.0, 1.0))
        # Do cong = do vong cua vach TRONG TAM NHIN: lech ngang giua diem xa
        # nhat quan sat duoc va tiep tuyen keo dai tu muc xe. Dau duong = cua phai.
        raw_curv = float(np.clip(coeff[0] * t_max * t_max / half, -1.0, 1.0))

        self._anchor = float(x_at(0.0))

        if not self._initialised:
            self._cte_ema, self._look_ema, self._curv_ema = raw_cte, raw_look, raw_curv
            self._initialised = True
        else:
            a = self.alpha
            self._cte_ema = a * raw_cte + (1.0 - a) * self._cte_ema
            self._look_ema = a * raw_look + (1.0 - a) * self._look_ema
            self._curv_ema = a * raw_curv + (1.0 - a) * self._curv_ema

        return LaneResult(True, self._cte_ema, self._curv_ema, n_pixels, warped,
                          cte_lookahead=self._look_ema, n_bands=len(points),
                          fit=coeff)

    def debug_process(self, frame_bgr):
        """Nhu process() nhung tra ve ca cac buoc trung gian, de tune nguong.

        Tach rieng khoi process() co chu dich: process() la duong nong chay 30 Hz
        tren xe, khong duoc ganh them chi phi dung debug.
        """
        small = frame_bgr
        if small.shape[1] != self.pw or small.shape[0] != self.ph:
            small = cv2.resize(small, (self.pw, self.ph))
        # Debug: van tinh mask CA khung de nhin thay cai gi bi ROI cat bo.
        binary = self._binarise(small)

        y0 = int(self.ph * self.roi_top)
        y1 = int(self.ph * self.roi_bottom)
        masked = np.zeros((self.ph, self.pw), np.uint8)
        if y1 > y0:
            masked[y0:y1, :] = self._binarise(small[y0:y1, :])
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
        self._look_ema = 0.0
        self._curv_ema = 0.0
        self._anchor = None
        self._initialised = False
