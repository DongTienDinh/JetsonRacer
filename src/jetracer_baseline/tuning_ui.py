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
from .control.corner import CornerController
from .control.driver import build_driver
from .control.pid import PID
from .perception.lane import LaneDetector
from .manual_collection import (
    DatasetSessionWriter, shape_steering, shape_throttle, slew_towards)
from .perception.shading import ShadingCorrector
from .recorder import FrameRecorder


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
    ('control.v_straight', 'Ga doan THANG', 0.00, 0.60, 0.01),
    ('control.v_corner', 'Ga khi vao CUA', 0.00, 0.40, 0.01),
    ('control.curve_enter', 'Nguong VAO cua', 0.05, 0.80, 0.01),
    ('control.curve_exit', 'Nguong RA cua', 0.02, 0.70, 0.01),
    ('control.curve_feedforward', 'Lai theo do cong', 0.0, 2.5, 0.05),
    ('control.corner_steer_gain', 'Boi lai khi cua', 1.0, 3.0, 0.05),
    ('control.steer_max', 'Lai toi da', 0.10, 1.00, 0.05),
    ('control.pid.kp', 'PID Kp', 0.0, 2.0, 0.05),
    ('control.pid.kd', 'PID Kd', 0.0, 1.0, 0.01),
    ('control.steer_lookahead_weight', 'Trong so diem ngam', 0.0, 1.0, 0.05),
    ('control.slowdown', 'Bo ga theo lech', 0.0, 1.0, 0.01),
    ('control.v_min', 'Ga toi thieu', 0.00, 0.40, 0.01),
    ('control.v_max', 'Tran ga (an toan)', 0.00, 0.80, 0.01),
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


# Ba che do lai. Tach ro de khong bao gio co chuyen "tuong dang tay ma xe tu chay".
MODE_STOP = 'DUNG'
MODE_MANUAL = 'TAY CAM'
MODE_AUTO = 'TU DONG'


class ControllerShaper(object):
    """Bien gia tri truc tay cam tho thanh lenh lai/ga.

    Tach khoi widget de test duoc: `widgets.Controller` chi ton tai trong
    Jupyter, con phep bien doi nay moi la cho de sai dau/sai gioi han.

    Dung lai dung cac ham da kiem chung trong manual_collection (deadzone, expo,
    slew) thay vi viet lai - hai duong lai tay khac nhau se cho ra hai bo dataset
    khong so sanh duoc voi nhau.
    """

    def __init__(self, cfg):
        self.reset_from_config(cfg)
        self._steer = 0.0
        self._throttle = 0.0

    def reset_from_config(self, cfg):
        c = cfg.get
        self.deadzone = float(c('manual.deadzone', 0.08))
        self.max_steering = float(c('manual.max_steering', 0.60))
        self.steering_expo = float(c('manual.steering_expo', 0.45))
        self.steering_slew_rate = float(c('manual.steering_slew_rate', 1.5))
        self.min_throttle = float(c('manual.min_throttle', 0.12))
        self.max_throttle = float(c('manual.max_throttle', 0.30))
        self.throttle_rise_rate = float(c('manual.throttle_rise_rate', 1.2))
        self.throttle_fall_rate = float(c('manual.throttle_fall_rate', 4.0))

    def reset(self):
        self._steer = 0.0
        self._throttle = 0.0

    def shape(self, steer_raw, throttle_raw, dt, invert_steering=False,
              invert_throttle=False, deadman_ok=True):
        """Tra ve (steer, throttle) da qua deadzone, expo va gioi han slew."""
        steer_in = -steer_raw if invert_steering else steer_raw
        target_steer = shape_steering(steer_in, self.deadzone,
                                      self.max_steering, self.steering_expo)
        throttle_in = -throttle_raw if invert_throttle else throttle_raw
        target_throttle = shape_throttle(throttle_in, self.deadzone,
                                         self.min_throttle, self.max_throttle)
        if not deadman_ok:
            target_throttle = 0.0

        dt = max(1e-3, min(0.10, float(dt)))
        self._steer = slew_towards(self._steer, target_steer,
                                   self.steering_slew_rate, dt)
        # Nha ga nhanh hon tang ga - giong manual_collection.
        if (self._throttle * target_throttle < 0.0
                or abs(target_throttle) < abs(self._throttle)):
            rate = self.throttle_fall_rate
        else:
            rate = self.throttle_rise_rate
        self._throttle = slew_towards(self._throttle, target_throttle, rate, dt)
        return self._steer, self._throttle

    @property
    def command(self):
        return self._steer, self._throttle


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
        # Giu nguyen trang thai che do/ga khi keo slider: tao lai controller se
        # dat ga ve 0 moi lan chinh mot num, xe giat lien tuc luc dang tune.
        if getattr(self, 'corner', None) is None:
            self.corner = CornerController(cfg)
        else:
            self.corner.reset_from_config(cfg)
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
        self.corner.reset()

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
        pid_out = self.pid.step(err, dt)
        steer, throttle, mode = self.corner.step(
            pid_out, res.cte, res.curvature, dt, lane_found=res.found)
        ramp = self.soft_start_scale(now)
        throttle *= ramp

        self.stats.push(res.cte, steer, res.found, res.n_bands)
        return {
            'proc': proc, 'lane': res, 'steer': steer, 'throttle': throttle,
            'error': err, 'ramp': ramp, 'mode': mode, 'pid': pid_out,
        }

    # ------------------------------------------------------------------ render
    def render_panel(self, result, fps=0.0, armed=False, width=None,
                     driver_kind='dryrun', ui_mode=None):
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
        panel = _draw_banner(panel, res, result, fps, armed, driver_kind,
                         ui_mode=ui_mode)
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


def _draw_banner(panel, res, result, fps, armed, driver_kind='dryrun',
                 ui_mode=None):
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
    if armed and ui_mode == MODE_MANUAL:
        # Phai phan biet ro: o che do nay CV van chay va van hien so lieu,
        # nhung nguoi moi la nguoi dieu khien xe.
        state = ('LAI TAY - ban dieu khien (CV chi do)' if live
                 else 'LAI TAY (DRYRUN - banh khong quay)')
        colour = (255, 160, 60)
    elif not armed:
        state, colour = 'DANG DUNG (bam CHAY hoac LAI TAY)', (160, 160, 160)
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

    # Che do lai o giua: nhin mot cai la biet vi sao ga dang cao hay thap.
    mode = result.get('mode')
    if mode:
        corner = (mode != 'THANG')
        cv2.putText(banner, mode, (int(width * 0.40), 44), font, 0.55,
                    (0, 165, 255) if corner else (120, 240, 120), 2)

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
                 soft_start_s=1.0, record_dir='logs', record_fps=15.0,
                 controller_index=0, data_root='data/driving',
                 control_hz=30.0):
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
        self.shaper = ControllerShaper(self.engine.cfg)
        self.controller = widgets.Controller(index=int(controller_index))
        self.data_root = data_root
        self.control_period = 1.0 / max(5.0, float(control_hz))

        self._grabber = None
        self._driver = None
        self._armed = False
        self._stop_event = threading.Event()
        self._thread = None
        self._driver_lock = threading.Lock()
        self._fps = 0.0
        self.record_dir = record_dir
        self.record_fps = float(record_fps)
        self._recorder = None
        self._record_lock = threading.Lock()
        # Vong dieu khien PHAI tach khoi vong camera. Do duoc tren xe: vong
        # camera + ve panel chi chay ~6 Hz; lai tay o 6 Hz thi khong dieu khien
        # duoc. Thread nay chay 30 Hz doc lap, chi doc lenh CV moi nhat.
        self._control_thread = None
        self._control_stop = threading.Event()
        self._state_lock = threading.Lock()
        self._cv_steer = 0.0
        self._cv_throttle = 0.0
        self._man_steer = 0.0
        self._man_throttle = 0.0
        self._stick_raw = (0.0, 0.0)
        self._deadman_ok = True
        self.mode = MODE_STOP
        self._writer = None
        self._writer_lock = threading.Lock()
        self._last_sample = 0.0

        self.preview = widgets.Image(format='jpeg', width=preview_width)
        self.status = widgets.HTML(value='<b>Trang thai:</b> chua mo camera')
        self.metrics = widgets.HTML(value='<i>chua co so lieu</i>')
        self.output = widgets.Output(
            layout=widgets.Layout(border='1px solid #ddd', height='140px',
                                  overflow_y='auto'))

        # --- cai dat tay cam ---
        cfgget = self.engine.get_param
        self.controller_view = widgets.HTML(
            value='<b>Tay cam:</b> chua ket noi')
        self.steering_axis = widgets.BoundedIntText(
            value=2, min=0, max=15, description='Truc lai:',
            layout=widgets.Layout(width='170px'))
        self.throttle_axis = widgets.BoundedIntText(
            value=1, min=0, max=15, description='Truc ga:',
            layout=widgets.Layout(width='170px'))
        self.invert_steering = widgets.Checkbox(
            value=False, description='Dao lai',
            layout=widgets.Layout(width='140px'))
        self.invert_throttle = widgets.Checkbox(
            value=True, description='Dao ga',
            layout=widgets.Layout(width='140px'))
        self.use_deadman = widgets.Checkbox(
            value=False, description='Bat dead-man',
            layout=widgets.Layout(width='170px'))
        self.deadman_button = widgets.BoundedIntText(
            value=4, min=0, max=15, description='Nut dead-man:',
            layout=widgets.Layout(width='190px'))
        self.session_name = widgets.Text(
            value='bai1_taycam', description='Session:',
            layout=widgets.Layout(width='300px'))
        self.sample_hz = widgets.BoundedFloatText(
            value=float(cfgget('manual.sample_hz', 10.0)), min=1.0, max=30.0,
            step=1.0, description='Anh/giay:',
            layout=widgets.Layout(width='180px'))

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
        self.btn_pad = widgets.Button(description='KET NOI TAY CAM',
                                      button_style='info',
                                      layout=widgets.Layout(width='190px'))
        self.btn_manual = widgets.Button(
            description='LAI TAY (thu data)', button_style='warning',
            layout=widgets.Layout(width='210px'),
            tooltip='Ban lai bang tay cam; CV van chay nen de so sanh')
        self.btn_data = widgets.Button(description='GHI DATA (train)',
                                       button_style='success',
                                       layout=widgets.Layout(width='190px'))
        self.btn_record = widgets.Button(description='GHI VIDEO',
                                         layout=widgets.Layout(width='150px'))
        self.btn_reset = widgets.Button(description='Xoa thong ke')
        self.btn_save = widgets.Button(description='LUU CONFIG',
                                       button_style='success')
        self.btn_close = widgets.Button(description='Dong')

        self.btn_open.on_click(self._on_open)
        self.btn_run.on_click(self._on_run)
        self.btn_halt.on_click(self._on_disarm)
        self.btn_stop.on_click(self._on_emergency)
        self.btn_pad.on_click(self._on_probe_controller)
        self.btn_manual.on_click(self._on_manual)
        self.btn_data.on_click(self._on_data)
        self.btn_record.on_click(self._on_record)
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

            # Cong bo lenh CV cho thread dieu khien. Vong camera KHONG con tu
            # gui lenh: no chay ~6 Hz tren xe, gui truc tiep thi lai bi giat.
            with self._state_lock:
                self._cv_steer = result['steer']
                self._cv_throttle = result['throttle']

            self._record_frame(frame, frame_id, result, now)
            self._record_dataset(frame, frame_id, result, now,
                                 time.monotonic())

            if (time.monotonic() - last_preview) >= 0.10:
                last_preview = time.monotonic()
                try:
                    panel = self.engine.render_panel(
                        result, fps=self._fps, armed=self._armed,
                        width=self.preview_width,
                        driver_kind=self.driver_kind, ui_mode=self.mode)
                    data = self.engine.encode_jpeg(panel)
                    if data:
                        self.preview.value = data
                    self.metrics.value = self._metrics_html()
                except Exception as exc:
                    self._log('Loi ve preview: %s' % exc)

        # Camera dung vi bat ky ly do gi -> khong duoc de xe chay tiep.
        self._disarm()
        with self._record_lock:
            if self._recorder is not None:
                self._log('Camera dung -> tu dong dong file ghi.')
                self._stop_record_locked()
        with self._writer_lock:
            if self._writer is not None:
                self._log('Camera dung -> tu dong dong session data.')
                self._stop_data_locked()
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

    # --------------------------------------------------- vong dieu khien 30 Hz
    def _controller_ready(self):
        return bool(getattr(self.controller, 'connected', False)) and \
            len(self.controller.axes) > max(int(self.steering_axis.value),
                                            int(self.throttle_axis.value))

    def _read_sticks(self):
        try:
            return (float(self.controller.axes[
                        int(self.steering_axis.value)].value),
                    float(self.controller.axes[
                        int(self.throttle_axis.value)].value))
        except Exception:
            return (0.0, 0.0)

    def _deadman_value(self):
        if not self.use_deadman.value:
            return True
        try:
            return bool(self.controller.buttons[
                int(self.deadman_button.value)].value)
        except Exception:
            # Khong doc duoc nut dead-man -> coi nhu KHONG giu. Mac dinh an toan
            # phai la cat ga, khong phai cho chay tiep.
            return False

    def _ensure_control_thread(self):
        if self._control_thread is not None and self._control_thread.is_alive():
            return
        self._control_stop.clear()
        self._control_thread = threading.Thread(target=self._control_loop,
                                                name='tune-control')
        self._control_thread.daemon = True
        self._control_thread.start()

    def _control_loop(self):
        """Gui lenh xuong driver o nhip co dinh, DOC LAP voi vong camera.

        Vi sao phai tach: vong camera con ve panel va ma hoa JPEG nen tren xe
        chi chay khoang 6 Hz. Lai tay o 6 Hz thi khong dieu khien duoc, va o
        che do tu dong thi lenh toi driver cung giat theo nhip ve anh.
        """
        t_prev = time.time()
        while not self._control_stop.is_set():
            t0 = time.time()
            dt = t0 - t_prev
            t_prev = t0

            mode = self.mode
            if mode == MODE_MANUAL:
                steer_raw, throttle_raw = self._read_sticks()
                deadman = self._deadman_value()
                steer, throttle = self.shaper.shape(
                    steer_raw, throttle_raw, dt,
                    invert_steering=bool(self.invert_steering.value),
                    invert_throttle=bool(self.invert_throttle.value),
                    deadman_ok=deadman)
                with self._state_lock:
                    self._stick_raw = (steer_raw, throttle_raw)
                    self._deadman_ok = deadman
                    self._man_steer = steer
                    self._man_throttle = throttle
            elif mode == MODE_AUTO:
                with self._state_lock:
                    steer = self._cv_steer
                    throttle = self._cv_throttle
            else:
                steer, throttle = 0.0, 0.0
                self.shaper.reset()

            if self._armed and self._driver is not None:
                with self._driver_lock:
                    try:
                        self._driver.set(steer, throttle)
                    except Exception as exc:
                        self._log('LOI DRIVER -> dung xe: %s' % exc)
                        self._disarm()

            sleep = self.control_period - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

    # ------------------------------------------------------------- ghi dataset
    # Cot phu ngoai schema chuan cua DatasetSessionWriter. `cv_*` la lenh CV
    # truyen thong TREN CUNG FRAME nguoi dang lai -> so sanh truc tiep duoc
    # "CV se lai the nao" voi "nguoi da lai the nao", khong can chay lai lan hai.
    DATA_EXTRA = ['cv_steer', 'cv_throttle', 'cte', 'cte_lookahead',
                  'curvature', 'drive_mode', 'n_bands', 'lane_found', 'ui_mode']

    def _on_data(self, _b=None):
        with self._writer_lock:
            if self._writer is not None:
                self._stop_data_locked()
                return
            if self._grabber is None:
                self._log('Mo camera truoc khi ghi data.')
                return
            try:
                self._writer = DatasetSessionWriter(
                    self.data_root, self.session_name.value,
                    metadata=self._data_metadata(),
                    extra_fields=list(self.DATA_EXTRA))
            except Exception as exc:
                self._writer = None
                self._log('Khong mo duoc session data: %s' % exc)
                return
            path = self._writer.session_dir
        self.btn_data.description = 'DUNG GHI DATA'
        self.btn_data.button_style = 'danger'
        self._log('BAT DAU GHI DATA: %s (%.0f anh/giay)'
                  % (path, float(self.sample_hz.value)))
        self._log('Nhan lai/ga cua NGUOI nam o steering_cmd/throttle_cmd; '
                  'lenh CV cung frame nam o cv_steer/cv_throttle.')

    def _stop_data_locked(self):
        writer = self._writer
        self._writer = None
        if writer is None:
            return
        try:
            writer.close()
        except Exception as exc:
            self._log('Loi dong session data: %s' % exc)
        self.btn_data.description = 'GHI DATA (train)'
        self.btn_data.button_style = 'success'
        if writer.error is not None:
            self._log('LOI GHI DATA: %s' % writer.error)
        self._log('DUNG GHI DATA: %d mau (vut %d) -> %s'
                  % (writer.count, writer.dropped, writer.session_dir))

    def _data_metadata(self):
        cfg = self.engine.cfg
        return {
            'source_kind': self.source_kind,
            'driver_kind': self.driver_kind,
            'ui_mode': self.mode,
            'proc_size': list(self.engine.proc_size),
            'camera': cfg.get('camera'),
            'lane': cfg.get('lane'),
            'control': cfg.get('control'),
            'shading_enabled': bool(cfg.get('camera.shading.enabled', False)),
            'note': ('Anh la frame THO truoc khi sua mau/resize. '
                     'steering_cmd/throttle_cmd = lenh NGUOI lai (nhan de train). '
                     'cv_* = lenh CV truyen thong tren cung frame (de so sanh).'),
        }

    def _record_dataset(self, frame, frame_id, result, now, now_mono):
        """Luu mot mau train. Thua nhip thi bo qua, khong bao gio block camera."""
        writer = self._writer
        if writer is None:
            return
        interval = 1.0 / max(1.0, float(self.sample_hz.value))
        if (now_mono - self._last_sample) < interval:
            return
        self._last_sample = now_mono
        res = result['lane']
        with self._state_lock:
            steer_raw, throttle_raw = self._stick_raw
            man_steer = self._man_steer
            man_throttle = self._man_throttle
            deadman = self._deadman_ok
        try:
            writer.write(
                frame, frame_id, now, now_mono,
                steer_raw, throttle_raw, man_steer, man_throttle,
                deadman, bool(getattr(self.controller, 'connected', False)),
                extra={
                    'cv_steer': '%.4f' % result['steer'],
                    'cv_throttle': '%.4f' % result['throttle'],
                    'cte': '%.4f' % res.cte,
                    'cte_lookahead': '%.4f' % res.cte_lookahead,
                    'curvature': '%.4f' % res.curvature,
                    'drive_mode': result.get('mode', ''),
                    'n_bands': '%d' % res.n_bands,
                    'lane_found': '1' if res.found else '0',
                    'ui_mode': self.mode,
                })
        except Exception as exc:
            self._log('LOI GHI DATA -> dung ghi: %s' % exc)
            with self._writer_lock:
                self._stop_data_locked()

    # ------------------------------------------------------------- ghi video
    # Ghi frame THO (truoc resize/sua mau) co chu dich: con chay lai detector
    # voi nguong khac tren dung doan duong do, va con hieu chuan lai shading.
    # Ghi anh da xu ly thi mat goc, khong lam lai duoc.
    RECORD_FIELDS = ['cte', 'cte_lookahead', 'curvature', 'steer', 'throttle',
                     'drive_mode', 'n_bands', 'lane_found', 'armed']

    def _on_record(self, _b=None):
        with self._record_lock:
            if self._recorder is not None:
                self._stop_record_locked()
                return
            if self._grabber is None:
                self._log('Mo camera truoc khi ghi.')
                return
            try:
                if not os.path.isdir(self.record_dir):
                    os.makedirs(self.record_dir)
                path = os.path.join(
                    self.record_dir,
                    'tune_%s.avi' % time.strftime('%Y%m%d_%H%M%S'))
                self._recorder = FrameRecorder(
                    path, fps=self.record_fps,
                    extra_fields=list(self.RECORD_FIELDS))
            except Exception as exc:
                self._recorder = None
                self._log('Khong mo duoc file ghi: %s' % exc)
                return
        self.btn_record.description = 'DUNG GHI'
        self.btn_record.button_style = 'danger'
        self._log('BAT DAU GHI: %s (+ .sidecar.csv co cte/lai/ga tung frame)'
                  % path)

    def _stop_record_locked(self):
        recorder = self._recorder
        self._recorder = None
        if recorder is None:
            return
        try:
            recorder.close()
        except Exception as exc:
            self._log('Loi dong file ghi: %s' % exc)
        self.btn_record.description = 'GHI VIDEO'
        self.btn_record.button_style = ''
        if recorder.error is not None:
            self._log('LOI GHI VIDEO: %s' % recorder.error)
        self._log('DUNG GHI: %d frame (vut %d) -> %s'
                  % (recorder.n_written, recorder.n_dropped, recorder.path))

    def _record_frame(self, frame, frame_id, result, now):
        """Day frame vao queue ghi. Khong bao gio block vong camera."""
        recorder = self._recorder
        if recorder is None:
            return
        res = result['lane']
        recorder.submit(frame, frame_id, timestamp=now, extra={
            'cte': '%.4f' % res.cte,
            'cte_lookahead': '%.4f' % res.cte_lookahead,
            'curvature': '%.4f' % res.curvature,
            'steer': '%.4f' % result['steer'],
            'throttle': '%.4f' % result['throttle'],
            'drive_mode': result.get('mode', ''),
            'n_bands': '%d' % res.n_bands,
            'lane_found': '1' if res.found else '0',
            'armed': '1' if self._armed else '0',
        })
        if recorder.error is not None:
            with self._record_lock:
                self._stop_record_locked()

    # -------------------------------------------------------- handler tay cam
    def _on_probe_controller(self, _b=None):
        """Doc trang thai tay cam. Phai bam/xoay can truoc thi trinh duyet moi
        gui su kien gamepad dau tien - day la hanh vi cua Gamepad API, khong
        phai loi ket noi."""
        self._ensure_control_thread()
        connected = bool(getattr(self.controller, 'connected', False))
        n_axes = len(self.controller.axes)
        n_buttons = len(self.controller.buttons)
        name = getattr(self.controller, 'name', '') or '(khong ten)'
        self._log('Tay cam: connected=%s, axes=%d, buttons=%d, name=%s'
                  % (connected, n_axes, n_buttons, name))
        if not connected:
            self.controller_view.value = (
                '<b>Tay cam:</b> <span style="color:#b00020">CHUA KET NOI</span>'
                ' - cam receiver, BAM/XOAY can mot cai roi bam lai nut nay')
            self._set_status('TAY CAM CHUA KET NOI')
            return
        need = max(int(self.steering_axis.value), int(self.throttle_axis.value))
        if n_axes <= need:
            self.controller_view.value = (
                '<b>Tay cam:</b> %s - chi co %d truc, khong du cho truc %d'
                % (name, n_axes, need))
            self._log('Chon lai so truc lai/ga cho dung tay cam nay.')
            return
        steer_raw, throttle_raw = self._read_sticks()
        self.controller_view.value = (
            '<b>Tay cam:</b> <span style="color:#0a7">%s</span> - %d truc, '
            '%d nut | lai[%d]=%+.2f ga[%d]=%+.2f'
            % (name, n_axes, n_buttons, int(self.steering_axis.value),
               steer_raw, int(self.throttle_axis.value), throttle_raw))
        self._set_status('Tay cam OK - bam LAI TAY de dieu khien')

    def _enter_mode(self, mode):
        """Doi che do lai. Tra ve True neu vao duoc."""
        if self._grabber is None:
            self._log('Bam MO CAMERA truoc.')
            return False
        if mode == MODE_MANUAL:
            if not self._controller_ready():
                self._log('TU CHOI LAI TAY: tay cam chua san sang. Bam KET NOI '
                          'TAY CAM va kiem tra so truc.')
                self._set_status('TAY CAM CHUA SAN SANG')
                return False
            steer_raw, throttle_raw = self._read_sticks()
            # Can chua ve giua ma ARM thi xe giat ngay khi nhan lenh dau tien.
            if abs(steer_raw) > 0.20 or abs(throttle_raw) > 0.20:
                self._log('TU CHOI LAI TAY: can gat chua ve giua '
                          '(lai=%+.2f, ga=%+.2f).' % (steer_raw, throttle_raw))
                self._set_status('Tha hai can ve giua roi bam lai')
                return False
        try:
            if self._driver is None:
                self._driver = build_driver(self.driver_kind, self.engine.cfg)
            with self._driver_lock:
                self._driver.stop()
        except Exception as exc:
            self._set_status('LOI DRIVER - xe khong chay')
            self._log('Khong mo duoc driver %s: %s' % (self.driver_kind, exc))
            return False

        self.shaper.reset_from_config(self.engine.cfg)
        self.shaper.reset()
        self._ensure_control_thread()
        self.mode = mode
        self._armed = True
        return True

    def _on_manual(self, _b=None):
        if self.mode == MODE_MANUAL:
            self._on_disarm()
            return
        if not self._enter_mode(MODE_MANUAL):
            return
        self.engine.stop_run()          # che do tay khong dung ramp cua CV
        live = (self.driver_kind != 'dryrun')
        self.btn_manual.description = 'DUNG LAI TAY'
        if live:
            self._set_status('DANG LAI TAY - ban dieu khien xe. CV van chay de '
                             'so sanh (khong dieu khien).')
        else:
            self._set_status('LAI TAY o DRYRUN - banh khong quay.')
        self._log('LAI TAY: driver=%s, lai toi da %.2f, ga toi da %.2f, '
                  'dead-man=%s' % (self.driver_kind, self.shaper.max_steering,
                                   self.shaper.max_throttle,
                                   'BAT' if self.use_deadman.value else 'tat'))

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
        if self.mode == MODE_MANUAL:
            self._log('Dang o che do LAI TAY. Bam DUNG LAI TAY truoc khi cho '
                      'CV dieu khien - hai che do khong duoc chay cung luc.')
            self._set_status('Dang LAI TAY - dung truoc khi chuyen TU DONG')
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
            self.mode = MODE_AUTO
            self._ensure_control_thread()
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
        self.mode = MODE_STOP
        self.engine.stop_run()
        self.shaper.reset()
        self.btn_manual.description = 'LAI TAY (thu data)'
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
            'ga ngay. Camera dung hoac loi xu ly -> tu dong cat ga.<br>'
            '<b>LAI TAY</b> = ban dieu khien, CV van chay nhung chi ghi so lieu. '
            '<b>CHAY</b> = CV dieu khien. Hai nut loai tru nhau.' % mode)
        row_drive = w.HBox([self.btn_open, self.btn_pad, self.btn_manual,
                            self.btn_run])
        row_stop = w.HBox([self.btn_halt, self.btn_stop, self.btn_data,
                           self.btn_record])
        row_tools = w.HBox([self.btn_reset, self.btn_save, self.btn_close])

        pad_box = w.VBox([
            self.controller_view,
            w.HBox([self.steering_axis, self.throttle_axis,
                    self.invert_steering, self.invert_throttle]),
            w.HBox([self.use_deadman, self.deadman_button]),
            w.HTML('<hr><b>Thu data de train model</b>'),
            w.HBox([self.session_name, self.sample_hz]),
            w.HTML(
                'Anh luu vao <code>data/driving/&lt;session&gt;_&lt;gio&gt;/</code>'
                ' gom <code>images/</code>, <code>labels.csv</code>,'
                ' <code>metadata.json</code>.<br>'
                'Nhan de train nam o <code>steering_cmd</code>/'
                '<code>throttle_cmd</code> (lenh NGUOI lai). Cot '
                '<code>cv_steer</code>/<code>cv_throttle</code> la lenh CV '
                'truyen thong TREN CUNG FRAME - so sanh duoc ngay CV lai the '
                'nao so voi nguoi, khong phai chay lai lan hai.'),
        ])

        tabs = w.Tab(children=[self.lane_box, self.control_box, pad_box])
        tabs.set_title(0, 'Bam vach')
        tabs.set_title(1, 'Dieu khien')
        tabs.set_title(2, 'Tay cam + Data')
        left = w.VBox([self.preview, self.status, self.metrics])
        right = w.VBox([w.HBox([self.line_color, self.use_shading]), tabs])
        return w.VBox([warning, row_drive, row_stop, row_tools,
                       w.HBox([left, right]), self.output])

    def show(self):
        from IPython.display import display
        display(self.widget())
        return self

    def close(self):
        self._stop_event.set()
        self._control_stop.set()
        self._disarm()
        if self._control_thread is not None and self._control_thread.is_alive():
            self._control_thread.join(timeout=2.0)
        with self._record_lock:
            if self._recorder is not None:
                self._stop_record_locked()
        with self._writer_lock:
            if self._writer is not None:
                self._stop_data_locked()
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
                     save_path='configs/tuned.yaml', soft_start_s=1.0,
                     controller_index=0, data_root='data/driving'):
    """Mo giao dien. Khong co lenh nao xuong phan cung cho den khi bam CHAY.

    `driver_kind=None` -> tu chon: replay video/anh tong hop thi khong the dieu
    khien xe that, nen dung dryrun; camera that thi dung nvidia.
    """
    if driver_kind is None:
        driver_kind = 'dryrun' if source_kind in ('video', 'synthetic') else 'nvidia'
    return LaneTuningUI(config_path=config_path, source_kind=source_kind,
                        video_path=video_path, driver_kind=driver_kind,
                        save_path=save_path, soft_start_s=soft_start_s,
                        controller_index=controller_index,
                        data_root=data_root).show()
