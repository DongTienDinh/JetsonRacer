# -*- coding: utf-8 -*-
"""Sua lens shading mau (color falloff) cua camera CSI.

VAN DE DO DUOC tren raw_camera.avi (trung binh 1485 frame):

    r (ban kinh chuan hoa)   R/G      B/G
    0.00-0.12                0.963    0.986     <- tam anh: trung tinh
    0.88-1.00                1.590    1.178     <- goc anh: do gap ~1.6 lan

Ban do R/G doi xung gan nhu hoan hao (trai 1.531 / phai 1.515, tren 1.364 /
duoi 1.337). Noi dung canh KHONG the tao ra hinh dang doi xung nhu vay -> day
la lens shading cua ong kinh, khong phai white balance sai va khong phai canh.

Vi sao phai sua truoc khi lam bat ky viec gi khac: bam lane tren sa ban nay
phai tach line DO theo mau (nguong xam bat nham vien trang - xem lane.py).
Neu goc anh do gap 1.6 lan tam anh thi mot nguong HSV duy nhat khong the dung
cho ca khung hinh: chinh du bat line o giua thi vien anh bao do gia, chinh du
sach vien thi mat line o giua.

CACH SUA: chi chuan hoa CHROMA (ti le R/G va B/G) theo ban kinh, GIU NGUYEN do
sang. Gain duoc chuan hoa sao cho tich ba kenh = 1 -> anh khong sang/toi di,
chi het am mau. Lam vay an toan hon sua vignette do sang: khong keo nhieu o
goc anh len va khong lam cháy vung da sang san.

Mo hinh: ti le do duoc theo ban kinh duoc fit bang da thuc chan (chuan cong
nghiep cho lens shading):

    ratio(r) = 1 + a1*r^2 + a2*r^4 + a3*r^6

He so nam trong configs/shading.yaml, sinh boi tools/calib_shading.py.
"""

import io
import os
import threading

import cv2
import numpy as np
import yaml


class ShadingCorrector(object):
    """Ap gain chroma theo ban kinh. Ban do gain duoc cache theo kich thuoc anh.

    Chi phi: hai phep nhan float32 tren moi pixel. Ban do gain duoc dung lai
    giua cac frame nen chi tinh mot lan cho moi kich thuoc.
    """

    def __init__(self, coeff_r, coeff_b, enabled=True, max_gain=3.0):
        # coeff_* = [a1, a2, a3] cua ratio(r) = 1 + a1*r^2 + a2*r^4 + a3*r^6
        self.coeff_r = [float(c) for c in coeff_r]
        self.coeff_b = [float(c) for c in coeff_b]
        self.enabled = bool(enabled)
        self.max_gain = float(max_gain)
        self._cache = {}
        # Buffer float dung lai giua cac frame. Doi voi 640x480 thi moi lan
        # `astype(np.float32)` cap phat 3.7 MB moi; tai dung giam 4.4 ms xuong
        # 2.5 ms/frame (do tren PC dev). Co khoa vi `apply()` co the bi goi tu
        # thread camera lan thread khac.
        self._scratch = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ config
    @classmethod
    def disabled(cls):
        return cls([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], enabled=False)

    @classmethod
    def from_config(cls, cfg):
        """Doc tu Config chinh; tra ve corrector da TAT neu chua hieu chuan.

        Mac dinh la TAT. Config phai noi ro `camera.shading.enabled: true` thi
        moi sua mau - khong tu bat chi vi tinh co co file configs/shading.yaml
        nam do. Sua mau am tham la loi kho truy vet nhat: moi nguong mau da tune
        deu lech ma khong ai biet vi sao.
        """
        if not bool(cfg.get('camera.shading.enabled', False)):
            return cls.disabled()

        coeff_r = cfg.get('camera.shading.coeff_r')
        coeff_b = cfg.get('camera.shading.coeff_b')
        if coeff_r is None or coeff_b is None:
            path = cfg.get('camera.shading.file', 'configs/shading.yaml')
            if not path or not os.path.exists(path):
                raise IOError(
                    'camera.shading.enabled = true nhung khong tim thay he so '
                    'hieu chuan (%s). Chay tools/calib_shading.py truoc, hoac '
                    'dat enabled: false.' % path)
            with io.open(path, 'r', encoding='utf-8') as fh:
                data = yaml.safe_load(fh) or {}
            coeff_r = data.get('coeff_r')
            coeff_b = data.get('coeff_b')
            if coeff_r is None or coeff_b is None:
                raise IOError('File hieu chuan thieu coeff_r/coeff_b: ' + path)

        return cls(coeff_r, coeff_b, enabled=True,
                   max_gain=float(cfg.get('camera.shading.max_gain', 3.0)))

    @classmethod
    def from_file(cls, path):
        with io.open(path, 'r', encoding='utf-8') as fh:
            data = yaml.safe_load(fh) or {}
        return cls(data['coeff_r'], data['coeff_b'],
                   enabled=bool(data.get('enabled', True)),
                   max_gain=float(data.get('max_gain', 3.0)))

    # ------------------------------------------------------------------ gain map
    @staticmethod
    def _ratio(r2, coeff):
        """ratio(r) = 1 + a1*r^2 + a2*r^4 + a3*r^6, nhan r2 = r^2."""
        a1, a2, a3 = coeff
        return 1.0 + a1 * r2 + a2 * r2 * r2 + a3 * r2 * r2 * r2

    def _build_maps(self, w, h):
        """Ban do gain cho kenh B va R (kenh G la moc, gain = 1 truoc chuan hoa)."""
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        xx = (np.arange(w, dtype=np.float32) - cx) / max(cx, 1e-6)
        yy = (np.arange(h, dtype=np.float32) - cy) / max(cy, 1e-6)
        # r chuan hoa ve 1.0 tai GOC anh - dung quy uoc voi luc hieu chuan
        r2 = (yy[:, None] ** 2 + xx[None, :] ** 2) / 2.0

        ratio_r = self._ratio(r2, self.coeff_r)
        ratio_b = self._ratio(r2, self.coeff_b)
        # Ti le do duoc phai duong; ratio <= 0 nghia la he so fit hong.
        ratio_r = np.clip(ratio_r, 1e-3, None)
        ratio_b = np.clip(ratio_b, 1e-3, None)

        g_r = 1.0 / ratio_r
        g_b = 1.0 / ratio_b
        g_g = np.ones_like(g_r)

        # Chuan hoa giu do sang: tich ba gain = 1 -> chi doi mau, khong doi sang.
        norm = np.cbrt(g_r * g_g * g_b)
        g_r /= norm
        g_g /= norm
        g_b /= norm

        m = self.max_gain
        return (np.clip(g_b, 0.0, m).astype(np.float32),
                np.clip(g_g, 0.0, m).astype(np.float32),
                np.clip(g_r, 0.0, m).astype(np.float32))

    def maps_for(self, w, h):
        key = (int(w), int(h))
        if key not in self._cache:
            self._cache[key] = self._build_maps(key[0], key[1])
        return self._cache[key]

    # ------------------------------------------------------------------- apply
    def _gain3(self, w, h):
        key = ('g3', int(w), int(h))
        if key not in self._cache:
            g_b, g_g, g_r = self.maps_for(w, h)
            self._cache[key] = np.ascontiguousarray(
                np.dstack([g_b, g_g, g_r]).astype(np.float32))
        return self._cache[key]

    def apply(self, bgr):
        """Tra ve anh BGR uint8 MOI da khu am mau vien.

        LUON tra mang moi, khong tra buffer dung chung: recorder va dataset
        writer xep frame vao hang doi roi ghi dia o thread khac, tra buffer
        dung chung se lam cac frame da xep hang bi ghi de bang frame sau.
        """
        if not self.enabled:
            return bgr
        h, w = bgr.shape[:2]
        gain = self._gain3(w, h)
        with self._lock:
            scratch = self._scratch.get((w, h))
            if scratch is None or scratch.shape != bgr.shape:
                scratch = np.empty(bgr.shape, np.float32)
                self._scratch[(w, h)] = scratch
            np.multiply(bgr, gain, out=scratch, casting='unsafe')
            np.clip(scratch, 0.0, 255.0, out=scratch)
            return scratch.astype(np.uint8)

    def apply_resized(self, bgr, size):
        """Resize ve `size` TRUOC roi moi sua mau - duong nong cua vong dieu khien.

        Sua o 640x480 ton ~3.7 ms/frame tren PC dev; resize xuong 320x240 truoc
        chi con ~1.6 ms (trong do resize 0.12 ms). Ket qua giong nhau vi gain la
        ham tron theo ban kinh, con ngan sach FPS tren Jetson Nano thi khong cho
        phep tra gia gap doi cho cung mot ket qua.
        """
        w, h = int(size[0]), int(size[1])
        if bgr.shape[1] != w or bgr.shape[0] != h:
            bgr = cv2.resize(bgr, (w, h))
        if not self.enabled:
            return bgr
        return self.apply(bgr)

    def describe(self):
        return 'ShadingCorrector(enabled=%s, coeff_r=%s, coeff_b=%s)' % (
            self.enabled, self.coeff_r, self.coeff_b)


def measure_ratio_profile(mean_bgr, n_bins=8):
    """Do ti le R/G, B/G theo ban kinh tren mot anh trung binh.

    Dung chung cho ca luc hieu chuan lan luc kiem tra sau khi sua, de hai ben
    khong the lech quy uoc ban kinh.
    """
    mean_bgr = mean_bgr.astype(np.float64)
    b, g, r = mean_bgr[:, :, 0], mean_bgr[:, :, 1], mean_bgr[:, :, 2]
    h, w = mean_bgr.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r2 = (((xx - cx) / max(cx, 1e-6)) ** 2 + ((yy - cy) / max(cy, 1e-6)) ** 2) / 2.0

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        sel = (r2 >= edges[i]) & (r2 < edges[i + 1])
        if sel.sum() < 50:
            continue
        gg = g[sel].mean()
        rows.append({
            'r2_lo': float(edges[i]),
            'r2_hi': float(edges[i + 1]),
            'r2_mid': float((edges[i] + edges[i + 1]) / 2.0),
            'n': int(sel.sum()),
            'rg': float(r[sel].mean() / max(gg, 1e-6)),
            'bg': float(b[sel].mean() / max(gg, 1e-6)),
        })
    return rows


def fit_coefficients(profile):
    """Fit ratio(r) = 1 + a1*r^2 + a2*r^4 + a3*r^6 theo binh phuong toi thieu.

    Ep ratio(0) = 1 bang cach fit tren (ratio - 1) khong co hang so tu do:
    tam anh la moc trung tinh theo dinh nghia.
    """
    x = np.array([p['r2_mid'] for p in profile], dtype=np.float64)
    wts = np.array([p['n'] for p in profile], dtype=np.float64)
    wts = np.sqrt(wts / wts.max())

    out = {}
    for key in ('rg', 'bg'):
        y = np.array([p[key] for p in profile], dtype=np.float64)
        # Chuan hoa ve tam: bin trong cung la moc 1.0
        y = y / y[0]
        design = np.stack([x, x ** 2, x ** 3], axis=1)
        coeff, _res, _rank, _sv = np.linalg.lstsq(
            design * wts[:, None], (y - 1.0) * wts, rcond=None)
        out[key] = [float(c) for c in coeff]
    return out['rg'], out['bg']
