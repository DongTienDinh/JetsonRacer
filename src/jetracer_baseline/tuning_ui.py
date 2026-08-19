# -*- coding: utf-8 -*-
"""Giao dien JupyterLab: xem camera truc tiep, chinh nguong bam vach, chay thu.

Muc dich: khong phai SSH vao Jetson go lenh nua. Mo notebook tu may khac, keo
slider, nhin ngay ket qua tren dung frame camera dang chay.

KIEN TRUC - hai lop tach roi co chu dich:

  LaneTuningEngine : toan bo phan xu ly. KHONG phu thuoc ipywidgets, chay duoc
                     tren laptop voi file video -> test duoc bang test_smoke.
  LaneTuningUI     : chi la lop widget mong noi slider vao engine.

Tach nhu vay vi phan de sai la xu ly anh va dieu khien, ma phan do lai la phan
khong test duoc neu bi tron vao widget.

AN TOAN - giu dung ky luat cua manual_collection.py:
  - Mac dinh `driver_kind='dryrun'`: keo slider bao nhieu cung khong quay banh.
  - Lenh chi ra driver sau khi bam ARM. Chua ARM thi steer/throttle chi hien thi.
  - Nut DUNG KHAN CAP cat ga ngay va DISARM.
  - Camera dung/loi -> tu dong DISARM.
  - Watchdog cua driver van chay: mat lenh qua `command_timeout_s` -> throttle=0.
"""

import io
import os
import threading
import time

import cv2
import numpy as np
import yaml

from .camera import LatestFrameGrabber, build_source, format_camera_environment
from .config import load_config
from .control.driver import build_driver
from .control.pid import PID
from .perception.lane import LaneDetector
from .perception.shading import ShadingCorrector


# Cac tham so hien ra slider: (duong dan config, nhan, min, max, buoc)
# Chi dua ra nhung tham so THUC SU can chinh tai sa ban. Slider cho moi khoa
# config se lam giao dien khong dung duoc trong 5 phut chuan bi.
LANE_PARAMS = [
    ('lane.roi_top', 'ROI tren', 0.30, 0.90, 0.01),
    ('lane.hsv_s_min', 'Bao hoa toi thieu (S)', 0, 255, 5),
    ('lane.hsv_v_min', 'Do sang toi thieu (V)', 0, 255, 5),
    ('lane.min_blob_area', 'Dien tich blob toi thieu', 1, 200, 1),
    ('lane.n_bands', 'So dai ngang', 4, 16, 1),
    ('lane.band_min_pixels', 'Pixel toi thieu / dai', 5, 200, 5),
    ('lane.min_bands', 'So dai toi thieu', 2, 6, 1),
    ('lane.max_run_frac', 'Be rong cum toi da', 0.10, 0.90, 0.01),
    ('lane.lookahead', 'Diem ngam xa', 0.10, 1.00, 0.05),
    ('lane.smooth_alpha', 'EMA alpha', 0.10, 1.00, 0.05),
]

CONTROL_PARAMS = [
    ('control.pid.kp', 'PID Kp', 0.0, 2.0, 0.05),
    ('control.pid.kd', 'PID Kd', 0.0, 1.0, 0.01),
    ('control.steer_lookahead_weight', 'Trong so diem ngam', 0.0, 1.0, 0.05),
    ('control.steer_max', 'Lai toi da', 0.10, 1.00, 0.05),
    ('control.v_max', 'Ga toi da', 0.00, 0.60, 0.01),
    ('control.v_min', 'Ga toi thieu', 0.00, 0.40, 0.01),
    ('control.slowdown', 'Bo ga theo lech', 0.0, 1.0, 0.01),
    ('control.curve_slowdown', 'Bo ga theo cua', 0.0, 1.5, 0.05),
]


class RollingStats(object):
    """Thong ke cua so truot - de bat loi, khong phai de bao cao ket qua.

    So lieu chinh thuc cho paper phai lay tu file CSV cua mot luot chay day du,
    khong phai tu cua so truot cua giao dien tune.
    """

    def __init__(self, window=200):
        self.window = int(window)
        self._cte = []
        self._steer = []
        self._found = []
        self._bands = []

    def push(self, cte, steer, found, bands):
        for buf, value in ((self._cte, cte), (self._steer, steer),
                           (self._found, 1.0 if found else 0.0),
                           (self._bands, bands)):
            buf.append(float(value))
            if len(buf) > self.window:
                buf.pop(0)

    def reset(self):
        del self._cte[:], self._steer[:], self._found[:], self._bands[:]

    @property
    def n(self):
        return len(self._cte)

    def summary(self):
        if not self._cte:
            return None
        cte = np.array(self._cte)
        steer = np.array(self._steer)
        sign = np.sign(steer)
        flips = int(np.sum(sign[1:] * sign[:-1] < 0)) if len(steer) > 1 else 0
        return {
            'n': len(cte),
            'cte_rms': float(np.sqrt(np.mean(cte ** 2))),
            'cte_p95': float(np.percentile(np.abs(cte), 95)),
            'loss_pct': float(100.0 * (1.0 - np.mean(self._found))),
            'bands_mean': float(np.mean(self._bands)),
            'steer_flips': flips,
            'steer_sat_pct': float(100.0 * np.mean(np.abs(steer) >= 0.999 * (
                np.max(np.abs(steer)) if np.max(np.abs(steer)) > 0 else 1.0))),
        }

    def cte_history(self):
        return list(self._cte)

    def steer_history(self):
        return list(self._steer)


class LaneTuningEngine(object):
    """Xu ly mot frame: sua mau -> bam vach -> sinh lenh -> ve panel debug."""

    def __init__(self, config_path='configs/default.yaml', overrides=None):
        self.config_path = config_path
        self.cfg = load_config(config_path, overrides or [])
        self.stats = RollingStats()
        # Ga khong duoc nhay tu 0 len v_max ngay khi bam CHAY: xe giat, banh
        # truot, va nguoi bam khong kip phan ung neu huong lai dang sai. Ramp
        # tuyen tinh trong `soft_start_s` giay dau.
        self.soft_start_s = 1.0
        self._run_t0 = None
        self._dirty = True
        self.rebuild()

    # ------------------------------------------------------------------ config
    def set_param(self, dotted_key, value):
        self.cfg.set(dotted_key, value)
        self._dirty = True

    def get_param(self, dotted_key, default=None):
        return self.cfg.get(dotted_key, default)

    def rebuild(self):
        """Dung lai detector/PID tu config hien tai. Re - chi tao ma tran warp."""
        cfg = self.cfg
        # `lane.hsv_s_min` / `lane.hsv_v_min` duoc LaneDetector doc truc tiep va
        # ap len preset cua mau dang chon -> UI khong can (va khong duoc) ghi
        # thang dai HSV vao config.
        self.shading = ShadingCorrector.from_config(cfg)
        self.lane = LaneDetector(cfg)
        self.pid = PID(cfg.get('control.pid.kp', 0.6),
                       cfg.get('control.pid.ki', 0.0),
                       cfg.get('control.pid.kd', 0.1),
                       out_limit=cfg.get('control.pid.out_limit', 1.0))
        self.proc_size = (int(cfg.get('pipeline.proc_width', 320)),
                          int(cfg.get('pipeline.proc_height', 240)))
        self.v_max = float(cfg.get('control.v_max', 0.20))
        self.v_min = float(cfg.get('control.v_min', 0.10))
        self.slowdown = float(cfg.get('control.slowdown', 0.12))
        self.curve_slowdown = float(cfg.get('control.curve_slowdown', 0.35))
        self.steer_look_w = float(cfg.get('control.steer_lookahead_weight', 0.5))
        self.steer_max = float(cfg.get('control.steer_max', 0.60))
        self._dirty = False

    # --------------------------------------------------------------- run state
    def start_run(self, now=None):
        """Danh dau bat dau chay -> ga bat dau ramp tu 0."""
        self._run_t0 = time.time() if now is None else now
        self.pid.reset()

    def stop_run(self):
        self._run_t0 = None

    @property
    def running(self):
        return self._run_t0 is not None

    def soft_start_scale(self, now=None):
        """He so ga 0..1. Bang 1 khi chua chay (de preview hien ga day du)."""
        if self._run_t0 is None:
            return 1.0
        if self.soft_start_s <= 0:
            return 1.0
        elapsed = (time.time() if now is None else now) - self._run_t0
        if elapsed >= self.soft_start_s:
            return 1.0
        return max(0.0, elapsed / self.soft_start_s)

    # ----------------------------------------------------------------- process
    def process(self, frame_bgr, dt, now=None):
        """Tra ve dict ket qua. KHONG gui lenh ra driver - do la viec cua UI."""
        if self._dirty:
            self.rebuild()

        proc = self.shading.apply_resized(frame_bgr, self.proc_size)
        res = self.lane.process(proc)

        err = ((1.0 - self.steer_look_w) * res.cte
               + self.steer_look_w * res.cte_lookahead)
        steer = self.pid.step(err, dt)
        steer = max(-self.steer_max, min(self.steer_max, steer))

        speed = (self.v_max
                 - self.slowdown * abs(res.cte)
                 - self.curve_slowdown * abs(res.curvature))
        throttle = max(self.v_min, min(self.v_max, speed))
        ramp = self.soft_start_scale(now)
        throttle *= ramp

        self.stats.push(res.cte, steer, res.found, res.n_bands)
        return {
            'proc': proc, 'lane': res, 'steer': steer, 'throttle': throttle,
            'error': err, 'ramp': ramp,
        }

    # ------------------------------------------------------------------ render
    def render_panel(self, result, fps=0.0, armed=False, width=None,
                     driver_kind='dryrun'):
        """Panel 2x2: anh+mask, mask nhi phan, bird's-eye+fit, do thi CTE."""
        res = result['lane']
        proc = result['proc']
        ph, pw = proc.shape[:2]

        y0 = int(ph * float(self.cfg.get('lane.roi_top', 0.55)))
        mask = np.zeros((ph, pw), np.uint8)
        if ph > y0:
            mask[y0:, :] = self.lane._binarise(proc[y0:, :])

        # (1) anh that + mask phu len + duong ROI
        overlay = proc.copy()
        overlay[mask > 0] = (0, 0, 255)
        cv2.line(overlay, (0, y0), (pw, y0), (0, 255, 255), 1)
        _label(overlay, 'CAMERA + MASK VACH')

        # (2) mask nhi phan - nhin thang xem nguong co sach khong
        binary = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        _label(binary, 'MASK (nguong mau)')

        # (3) bird's-eye + duong fit + diem ngam
        warped = res.debug
        if warped is None:
            warped = np.zeros((ph, pw), np.uint8)
        bev = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        cv2.line(bev, (pw // 2, 0), (pw // 2, ph), (0, 255, 0), 1)
        if res.fit is not None:
            pts = []
            for t in np.linspace(0.0, 1.0, 41):
                x = res.fit[0] * t * t + res.fit[1] * t + res.fit[2]
                pts.append([int(x), ph - int(t * ph)])
            cv2.polylines(bev, [np.array(pts, np.int32)], False, (0, 0, 255), 2)
            t_look = float(self.cfg.get('lane.lookahead', 0.6))
            xl = res.fit[0] * t_look * t_look + res.fit[1] * t_look + res.fit[2]
            cv2.circle(bev, (int(xl), ph - int(t_look * ph)), 5, (255, 0, 255), -1)
        _label(bev, "BIRD'S-EYE + duong fit")

        # (4) do thi CTE/lai truot - cho thay dao dong, thu bang so khong thay
        chart = _draw_chart(self.stats.cte_history(), self.stats.steer_history(),
                            pw, ph)

        top = np.hstack([overlay, binary])
        bottom = np.hstack([bev, chart])
        panel = np.vstack([top, bottom])
        panel = _draw_banner(panel, res, result, fps, armed, driver_kind)
        if width:
            scale = float(width) / panel.shape[1]
            panel = cv2.resize(panel, (int(width), int(panel.shape[0] * scale)))
        return panel

    def encode_jpeg(self, image, quality=75):
        ok, buf = cv2.imencode('.jpg', image,
                               [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return buf.tobytes() if ok else None

    # -------------------------------------------------------------------- save
    def save_overrides(self, path):
        """Ghi RIENG cac tham so da doi so voi file config goc.

        Ghi de nguyen file default.yaml se mat toan bo comment giai thich vi sao
        tung con so duoc chon - phan dat gia nhat cua file do. File override nho
        dung duoc voi `--override` va doc duoc trong diff.
        """
        base = load_config(self.config_path).as_dict()
        current = self.cfg.as_dict()
        diff = _dict_diff(base, current)
        if not diff:
            return None
        directory = os.path.dirname(path)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
        header = (
            u'# Sinh boi giao dien tune (src/jetracer_baseline/tuning_ui.py)\n'
            u'# %s\n'
            u'# Dung: python3 -m src.jetracer_baseline.cli run --task speed '
            u'--driver nvidia --override %s\n'
            u'# CHI chua tham so KHAC voi %s.\n\n'
            % (time.strftime('%Y-%m-%d %H:%M:%S'), path, self.config_path))
        with io.open(path, 'w', encoding='utf-8') as fh:
            fh.write(header)
            fh.write(yaml.safe_dump(diff, default_flow_style=False,
                                    allow_unicode=True))
        return diff


# --------------------------------------------------------------------- helpers
def _label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 16), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (255, 255, 255), 1)


def _draw_chart(cte_hist, steer_hist, w, h):
    """Do thi truot: cte (vang) va lai (xanh), thang do co dinh [-1, 1]."""
    chart = np.zeros((h, w, 3), np.uint8)
    chart[:] = (24, 24, 24)
    mid = h // 2
    cv2.line(chart, (0, mid), (w, mid), (70, 70, 70), 1)
    for frac in (0.25, 0.75):
        y = int(h * frac)
        cv2.line(chart, (0, y), (w, y), (45, 45, 45), 1)

    def plot(values, colour):
        if len(values) < 2:
            return
        n = min(len(values), w)
        vals = values[-n:]
        step = float(w) / max(1, n - 1)
        pts = [[int(i * step), int(mid - np.clip(v, -1.0, 1.0) * (h / 2 - 4))]
               for i, v in enumerate(vals)]
        cv2.polylines(chart, [np.array(pts, np.int32)], False, colour, 1)

    plot(steer_hist, (120, 220, 120))
    plot(cte_hist, (60, 220, 240))
    _label(chart, 'CTE (vang) / LAI (xanh)  thang do +-1')
    return chart


def _draw_banner(panel, res, result, fps, armed, driver_kind='dryrun'):
    """Bang trang thai tren cung panel.

    Khoi so lieu duoc CANH PHAI theo be rong do duoc, khong dat o mot ty le co
    dinh: chuoi trang thai dai ngan khac nhau tuy driver, dat cung cho thi hai
    khoi chu de len nhau va khong doc duoc con so nao.
    """
    width = panel.shape[1]
    banner = np.zeros((58, width, 3), np.uint8)
    banner[:] = (18, 18, 18)
    font = cv2.FONT_HERSHEY_SIMPLEX

    live = (str(driver_kind) != 'dryrun')
    ramp = float(result.get('ramp', 1.0))
    if not armed:
        state, colour = 'DANG DUNG (bam CHAY de xe di)', (160, 160, 160)
    elif not live:
        # Vang = dryrun khong gui gi xuong phan cung, banh dung yen. Bao "xe dang
        # chay" o day la noi doi va nguoi dung se ngoi doi xe chay.
        state, colour = 'DRYRUN - BANH KHONG QUAY', (0, 200, 255)
    elif ramp < 1.0:
        state, colour = 'DANG CHAY - tang ga %d%%' % int(ramp * 100), (0, 200, 255)
    else:
        state, colour = 'DANG CHAY - BAM LINE', (60, 60, 255)

    cv2.putText(banner, state, (8, 20), font, 0.5, colour, 2)
    found = 'CO' if res.found else 'MAT VACH'
    cv2.putText(banner, 'vach=%s  dai=%d  FPS(UI)=%.1f  driver=%s'
                % (found, res.n_bands, fps, driver_kind),
                (8, 44), font, 0.42,
                (120, 240, 120) if res.found else (60, 60, 255), 1)

    right = [
        'cte=%+.3f  ngam=%+.3f  cong=%+.3f' % (
            res.cte, res.cte_lookahead, res.curvature),
        'lai=%+.3f  ga=%.3f' % (result['steer'], result['throttle']),
    ]
    widest = max(cv2.getTextSize(t, font, 0.42, 1)[0][0] for t in right)
    x = max(8, width - widest - 10)
    for i, text in enumerate(right):
        cv2.putText(banner, text, (x, 20 + i * 24), font, 0.42,
                    (240, 240, 240), 1)
    return np.vstack([banner, panel])


def _dict_diff(base, current):
    """Cac khoa trong `current` khac `base`, giu nguyen cau truc long nhau."""
    out = {}
    for key, value in current.items():
        if isinstance(value, dict):
            sub_base = base.get(key) if isinstance(base, dict) else None
            sub = _dict_diff(sub_base if isinstance(sub_base, dict) else {}, value)
            if sub:
                out[key] = sub
        else:
            if not isinstance(base, dict) or key not in base or base[key] != value:
                out[key] = value
    return out


# ============================================================== lop widget UI
class LaneTuningUI(object):
    """Giao dien Jupyter noi slider vao LaneTuningEngine."""

    def __init__(self, config_path='configs/default.yaml', source_kind='csi',
                 video_path=None, driver_kind='nvidia',
                 save_path='configs/tuned.yaml', preview_width=760,
                 soft_start_s=1.0):
        try:
            import ipywidgets.widgets as widgets
        except ImportError:
            raise ImportError(
                'Can ipywidgets trong moi truong Jupyter. Kiem tra bang: '
                'python3 -c "import ipywidgets; print(ipywidgets.__version__)"')

        # Giong manual_collection: uu tien do phan hoi cua vong dieu khien hon
        # thong luong tong khi thread camera dang ma hoa JPEG.
        import sys as _sys
        _sys.setswitchinterval(0.001)

        self.widgets = widgets
        self.source_kind = source_kind
        self.video_path = video_path
        self.driver_kind = driver_kind
        self.save_path = save_path
        self.preview_width = preview_width

        self.engine = LaneTuningEngine(config_path)
        self.engine.soft_start_s = float(soft_start_s)

        self._grabber = None
        self._driver = None
        self._armed = False
        self._stop_event = threading.Event()
        self._thread = None
        self._driver_lock = threading.Lock()
        self._fps = 0.0

        self.preview = widgets.Image(format='jpeg', width=preview_width)
        self.status = widgets.HTML(value='<b>Trang thai:</b> chua mo camera')
        self.metrics = widgets.HTML(value='<i>chua co so lieu</i>')
        self.output = widgets.Output(
            layout=widgets.Layout(border='1px solid #ddd', height='140px',
                                  overflow_y='auto'))

        self.line_color = widgets.Dropdown(
            options=[('do (sa ban tap)', 'red'), ('trang (sa ban thi)', 'white')],
            value=str(self.engine.get_param('lane.line_color', 'red')),
            description='Mau vach:')
        self.use_shading = widgets.Checkbox(
            value=bool(self.engine.get_param('camera.shading.enabled', False)),
            description='Sua mau (shading)')

        self._sliders = {}
        self.lane_box = self._build_sliders(LANE_PARAMS)
        self.control_box = self._build_sliders(CONTROL_PARAMS)

        self.btn_open = widgets.Button(description='1. MO CAMERA',
                                       button_style='info')
        run_label = ('2. CHAY (DRYRUN - banh khong quay)'
                     if driver_kind == 'dryrun' else '2. CHAY - BAM LINE')
        self.btn_run = widgets.Button(
            description=run_label, button_style='danger',
            layout=widgets.Layout(width='260px'),
            tooltip='Xe bat dau tu bam line. Ga tang dan trong %.1f giay dau.'
                    % soft_start_s)
        self.btn_halt = widgets.Button(description='DUNG',
                                       layout=widgets.Layout(width='120px'))
        self.btn_stop = widgets.Button(description='DUNG KHAN CAP',
                                       button_style='danger',
                                       layout=widgets.Layout(width='170px'))
        self.btn_reset = widgets.Button(description='Xoa thong ke')
        self.btn_save = widgets.Button(description='LUU CONFIG',
                                       button_style='success')
        self.btn_close = widgets.Button(description='Dong')

        self.btn_open.on_click(self._on_open)
        self.btn_run.on_click(self._on_run)
        self.btn_halt.on_click(self._on_disarm)
        self.btn_stop.on_click(self._on_emergency)
        self.btn_reset.on_click(lambda _b: self.engine.stats.reset())
        self.btn_save.on_click(self._on_save)
        self.btn_close.on_click(lambda _b: self.close())

        self.line_color.observe(self._on_colour_change, names='value')
        self.use_shading.observe(self._on_shading_change, names='value')

    # ------------------------------------------------------------------ helpers
    def _build_sliders(self, params):
        w = self.widgets
        rows = []
        for key, label, lo, hi, step in params:
            current = self.engine.get_param(key)
            if current is None:
                current = self._default_for(key, lo, hi)
            if isinstance(step, int) and float(step).is_integer() and \
                    isinstance(lo, int):
                slider = w.IntSlider(
                    value=int(current), min=int(lo), max=int(hi), step=int(step),
                    description=label, continuous_update=False,
                    style={'description_width': '160px'},
                    layout=w.Layout(width='420px'))
            else:
                slider = w.FloatSlider(
                    value=float(current), min=float(lo), max=float(hi),
                    step=float(step), description=label, readout_format='.2f',
                    continuous_update=False,
                    style={'description_width': '160px'},
                    layout=w.Layout(width='420px'))
            slider.observe(self._make_setter(key), names='value')
            self._sliders[key] = slider
            rows.append(slider)
        return w.VBox(rows)

    def _default_for(self, key, lo, hi):
        from .perception.lane import COLOR_PRESETS
        preset = COLOR_PRESETS.get(
            str(self.engine.get_param('lane.line_color', 'red')),
            COLOR_PRESETS['red'])
        if key == 'lane.hsv_s_min':
            return preset['hsv_low_1'][1]
        if key == 'lane.hsv_v_min':
            return preset['hsv_low_1'][2]
        if key == 'lane.min_blob_area':
            return preset['min_blob_area']
        return (lo + hi) / 2.0

    def _make_setter(self, key):
        def handler(change):
            self.engine.set_param(key, change['new'])
        return handler

    def _on_colour_change(self, change):
        self.engine.set_param('lane.line_color', change['new'])
        # Nguong S/V phu thuoc mau vach -> keo slider ve mac dinh cua preset moi,
        # neu khong nguoi dung se tune tiep tu nguong cua mau cu ma khong biet.
        for key in ('lane.hsv_s_min', 'lane.hsv_v_min', 'lane.min_blob_area'):
            if key in self._sliders:
                default = self._default_for(key, 0, 255)
                self._sliders[key].value = default
                self.engine.set_param(key, default)
        self._log('Doi mau vach sang "%s"; nguong S/V/blob da ve mac dinh preset.'
                  % change['new'])

    def _on_shading_change(self, change):
        self.engine.set_param('camera.shading.enabled', bool(change['new']))
        try:
            self.engine.rebuild()
        except IOError as exc:
            self.use_shading.value = False
            self.engine.set_param('camera.shading.enabled', False)
            self._log('Khong bat duoc sua mau: %s' % exc)

    def _log(self, message):
        with self.output:
            print('[%s] %s' % (time.strftime('%H:%M:%S'), message))

    def _set_status(self, message):
        self.status.value = '<b>Trang thai:</b> %s' % message

    # -------------------------------------------------------------------- loop
    def _loop(self, grabber):
        last_id = -1
        last_preview = 0.0
        t_prev = time.time()
        fps_t0 = time.monotonic()
        fps_n = 0

        while not self._stop_event.is_set():
            frame, frame_id = grabber.read()
            if frame is None or frame_id == last_id:
                if getattr(grabber, 'eof', False) or grabber.error is not None:
                    if grabber.error is not None:
                        self._log('CAMERA DUNG DO LOI: %s' % grabber.error)
                    else:
                        self._log('Nguon camera/video da het.')
                    break
                time.sleep(0.005)
                continue
            last_id = frame_id

            now = time.time()
            dt = now - t_prev
            t_prev = now
            fps_n += 1
            elapsed = time.monotonic() - fps_t0
            if elapsed >= 1.0:
                self._fps = fps_n / elapsed
                fps_t0 = time.monotonic()
                fps_n = 0

            try:
                result = self.engine.process(frame, dt, now=now)
            except Exception as exc:
                self._log('LOI XU LY -> DISARM: %s' % exc)
                self._disarm()
                break

            if self._armed and self._driver is not None:
                with self._driver_lock:
                    try:
                        self._driver.set(result['steer'], result['throttle'])
                    except Exception as exc:
                        self._log('LOI DRIVER -> DISARM: %s' % exc)
                        self._disarm()

            if (time.monotonic() - last_preview) >= 0.10:
                last_preview = time.monotonic()
                try:
                    panel = self.engine.render_panel(
                        result, fps=self._fps, armed=self._armed,
                        width=self.preview_width,
                        driver_kind=self.driver_kind)
                    data = self.engine.encode_jpeg(panel)
                    if data:
                        self.preview.value = data
                    self.metrics.value = self._metrics_html()
                except Exception as exc:
                    self._log('Loi ve preview: %s' % exc)

        # Camera dung vi bat ky ly do gi -> khong duoc de xe chay tiep.
        self._disarm()
        try:
            grabber.stop()
        except Exception:
            pass
        if self._grabber is grabber:
            self._grabber = None
        self._set_status('camera da dung; bam MO CAMERA de mo lai')

    def _metrics_html(self):
        s = self.engine.stats.summary()
        if s is None:
            return '<i>chua co so lieu</i>'
        warn = []
        if s['loss_pct'] > 2.0:
            warn.append('mat vach %.1f%% (muc tieu <= 2%%)' % s['loss_pct'])
        if s['bands_mean'] < 3.0:
            warn.append('chi %.1f dai - it du lieu, giam "Dien tich blob"'
                        % s['bands_mean'])
        if s['cte_rms'] > 0.15:
            warn.append('cte_rms %.3f (muc tieu <= 0.15)' % s['cte_rms'])
        rate = s['steer_flips'] / max(1.0, s['n'] / max(1.0, self._fps or 20.0))
        if rate > 3.0:
            warn.append('lai doi dau %.1f lan/giay - giam Kp hoac trong so ngam'
                        % rate)
        banner = ''
        if warn:
            banner = ('<div style="color:#b00020"><b>Canh bao:</b> %s</div>'
                      % '; '.join(warn))
        return (
            '%s<table style="font-family:monospace">'
            '<tr><td>so mau</td><td><b>%d</b></td>'
            '<td style="padding-left:18px">mat vach</td><td><b>%.1f%%</b></td></tr>'
            '<tr><td>cte_rms</td><td><b>%.3f</b></td>'
            '<td style="padding-left:18px">|cte| p95</td><td><b>%.3f</b></td></tr>'
            '<tr><td>dai TB</td><td><b>%.1f</b></td>'
            '<td style="padding-left:18px">lai doi dau</td><td><b>%d</b></td></tr>'
            '</table>' % (banner, s['n'], s['loss_pct'], s['cte_rms'],
                          s['cte_p95'], s['bands_mean'], s['steer_flips']))

    # ----------------------------------------------------------------- buttons
    def _on_open(self, _b=None):
        if self._grabber is not None:
            self._log('Camera dang mo.')
            return
        try:
            self._log('Dang mo camera kind=%s, %s'
                      % (self.source_kind, format_camera_environment()))
            source = build_source(self.engine.cfg, self.source_kind,
                                  video_path=self.video_path)
            grabber = LatestFrameGrabber(source).start()
            if grabber.error is not None:
                raise RuntimeError(grabber.error)
        except Exception as exc:
            self._set_status('KHONG MO DUOC CAMERA')
            self._log('Loi mo camera: %s' % exc)
            return

        self._grabber = grabber
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, args=(grabber,))
        self._thread.daemon = True
        self._thread.start()
        self._set_status('camera dang chay (chua ARM - xe khong chay)')
        self._log('Camera OK, backend=%s'
                  % getattr(source, 'backend', self.source_kind))

    def _on_run(self, _b=None):
        if self._grabber is None:
            self._log('Bam MO CAMERA truoc.')
            return
        if self._armed:
            self._log('Xe dang chay roi.')
            return
        summary = self.engine.stats.summary()
        # Khong cho ARM khi detector dang mat vach lien tuc: xe se lao theo gia
        # tri cte cu. Day la loi de xay ra nhat khi vua keo slider xong.
        if summary is not None and summary['n'] >= 20 and summary['loss_pct'] > 20.0:
            self._log('TU CHOI CHAY: dang mat vach %.0f%% frame. Xe se lao theo '
                      'gia tri cte cu. Chinh nguong cho mask sach roi bam lai.'
                      % summary['loss_pct'])
            self._set_status('TU CHOI CHAY - mask chua on')
            return
        try:
            if self._driver is None:
                self._driver = build_driver(self.driver_kind, self.engine.cfg)
            with self._driver_lock:
                self._driver.stop()
                self._armed = True
            self.engine.pid.reset()
            self.engine.start_run()
            if self.driver_kind == 'dryrun':
                self._set_status('DANG CHAY o che do DRYRUN - BANH KHONG QUAY '
                                 '(dung de xem thu, khong dieu khien xe)')
                self._log('CHAY (dryrun): khong co lenh nao xuong phan cung.')
            else:
                self._set_status('DANG CHAY - xe tu bam line (driver=%s). '
                                 'Ga tang dan trong %.1f giay dau.'
                                 % (self.driver_kind, self.engine.soft_start_s))
                self._log('CHAY: driver=%s, ga ramp %.1fs. Bam DUNG hoac DUNG '
                          'KHAN CAP de cat ga.'
                          % (type(self._driver).__name__,
                             self.engine.soft_start_s))
        except Exception as exc:
            self._armed = False
            self.engine.stop_run()
            self._set_status('LOI DRIVER - xe khong chay')
            self._log('Khong chay duoc voi driver %s: %s' % (self.driver_kind, exc))

    def _disarm(self):
        self._armed = False
        self.engine.stop_run()
        if self._driver is not None:
            with self._driver_lock:
                try:
                    self._driver.stop()
                except Exception:
                    pass

    def _on_disarm(self, _b=None):
        self._disarm()
        self._set_status('DA DUNG - motor=0')
        self._log('Da dung xe.')

    def _on_emergency(self, _b=None):
        self._disarm()
        self._set_status('DUNG KHAN CAP - motor=0, steering=0')
        self._log('DUNG KHAN CAP.')

    def _on_save(self, _b=None):
        try:
            diff = self.engine.save_overrides(self.save_path)
        except Exception as exc:
            self._log('Loi ghi config: %s' % exc)
            return
        if not diff:
            self._log('Khong co tham so nao khac config goc - khong ghi file.')
            return
        self._log('Da ghi %s. Chay that bang:' % self.save_path)
        self._log('  python3 -m src.jetracer_baseline.cli run --task speed '
                  '--driver nvidia --override %s' % self.save_path)

    # -------------------------------------------------------------------- view
    def widget(self):
        w = self.widgets
        if self.driver_kind == 'dryrun':
            mode = ('<b style="color:#0a7">DANG O CHE DO DRYRUN</b> - banh se '
                    'KHONG quay du co bam ARM. Dung de chinh nguong an toan. '
                    'Muon xe chay that: <code>ui.close()</code> roi '
                    '<code>launch(driver_kind="nvidia")</code>.')
        else:
            mode = ('<b style="color:#b00020">DRIVER=%s - XE SE CHAY THAT khi '
                    'bam ARM.</b> Ke banh khoi mat dat o lan ARM dau tien.'
                    % self.driver_kind)
        warning = w.HTML(
            '%s<br><b style="color:#b00020">AN TOAN:</b> nut DUNG KHAN CAP cat '
            'ga ngay. Camera dung hoac loi xu ly -> tu dong DISARM.' % mode)
        buttons = w.HBox([self.btn_open, self.btn_run, self.btn_halt,
                          self.btn_stop])
        tools = w.HBox([self.btn_reset, self.btn_save, self.btn_close])
        tabs = w.Tab(children=[self.lane_box, self.control_box])
        tabs.set_title(0, 'Bam vach')
        tabs.set_title(1, 'Dieu khien')
        left = w.VBox([self.preview, self.status, self.metrics])
        right = w.VBox([w.HBox([self.line_color, self.use_shading]), tabs])
        return w.VBox([warning, buttons, tools, w.HBox([left, right]),
                       self.output])

    def show(self):
        from IPython.display import display
        display(self.widget())
        return self

    def close(self):
        self._stop_event.set()
        self._disarm()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._grabber is not None:
            try:
                self._grabber.stop()
            except Exception:
                pass
            self._grabber = None
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception:
                pass
            self._driver = None
        self._log('Da dong giao dien tune.')


def launch_tuning_ui(config_path='configs/default.yaml', source_kind='csi',
                     video_path=None, driver_kind=None,
                     save_path='configs/tuned.yaml', soft_start_s=1.0):
    """Mo giao dien. Khong co lenh nao xuong phan cung cho den khi bam CHAY.

    `driver_kind=None` -> tu chon: replay video/anh tong hop thi khong the dieu
    khien xe that, nen dung dryrun; camera that thi dung nvidia.
    """
    if driver_kind is None:
        driver_kind = 'dryrun' if source_kind in ('video', 'synthetic') else 'nvidia'
    return LaneTuningUI(config_path=config_path, source_kind=source_kind,
                        video_path=video_path, driver_kind=driver_kind,
                        save_path=save_path, soft_start_s=soft_start_s).show()
