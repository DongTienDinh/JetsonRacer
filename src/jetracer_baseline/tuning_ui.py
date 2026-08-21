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
from collections import deque

import cv2
import numpy as np
import yaml

from .camera import (LatestFrameGrabber, build_source,
                     format_camera_environment, shading_applied_at_source)
from .config import load_config
from .control.corner import CornerController
from .control.driver import servo_output_for
from .control.driver import build_driver
from .control.pid import PID
from .perception import build_lane_detector
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

# Khi chay CNN, gan het LANE_PARAMS o tren la nguong MAU -> vo tac dung, vi
# model khong dung nguong mau. De nguyen chi lam nguoi tune keo nham roi tuong
# minh dang chinh gi do. Danh sach nay la nhung thu THAT SU con tac dung.
CNN_LANE_PARAMS = [
    ('lane.cnn.lookahead', 'Diem ngam xa', 0.10, 1.00, 0.05),
    ('lane.smooth_alpha', 'EMA alpha (do muot)', 0.10, 1.00, 0.05),
    ('lane.cnn.min_bands', 'So dai toi thieu', 2, 6, 1),
    ('lane.cnn.band_min_pixels', 'Pixel toi thieu / dai', 4, 60, 2),
    ('lane.cnn.reg_disagree', 'Nguong bat dong y seg/reg', 0.10, 1.00, 0.05),
]

# Che do DON GIAN: chi nhung num anh huong truc tiep den "xe chay nhanh cham va
# om cua the nao". Moi thu khac (nguong mau, tay cam, thu data, ghi video) bi an
# di - khong xoa khoi code, chi khong hien.
# Che do DON GIAN. Moi num kem MOT DONG giai thich hien ngay duoi slider -
# khong dung tooltip, vi tooltip phai ro chuot vao moi thay va nguoi tune dang
# cui xuong nhin xe chu khong nhin man hinh.
# Dinh dang: (khoa, nhan, min, max, buoc, giai thich)
# Che do DON GIAN: DUNG HAI NUM. Moi num dieu khien mot NHOM tham so, vi bat
# nguoi tune vặn ba cho de doi mot thu la cach chac chan lam ho vặn thieu mot cho.
#   TOC DO       -> v_straight, v_corner, v_max
#   GOC LAI TOI DA -> steering_output_max/min (tran CO KHI, la cai that su chan)
# Cac gain cua (feedforward, corner_steer_gain, curve_enter, kp) da duoc dat
# manh san trong configs/cnn.yaml - nguoi dung khong phai dong toi.
SIMPLE_PARAMS = []

CONTROL_PARAMS = [
    ('control.v_straight', 'Ga doan THANG', 0.00, 0.60, 0.01),
    ('control.v_corner', 'Ga khi vao CUA', 0.00, 0.40, 0.01),
    ('control.curve_enter', 'Nguong VAO cua', 0.05, 0.80, 0.01),
    ('control.curve_exit', 'Nguong RA cua', 0.02, 0.70, 0.01),
    ('control.curve_feedforward', 'Lai theo do cong', 0.0, 2.5, 0.05),
    ('control.corner_steer_gain', 'Boi lai khi cua', 1.0, 3.0, 0.05),
    ('control.steer_max', 'Lai toi da (thang)', 0.10, 1.00, 0.05),
    ('control.corner_steer_max', 'Lai toi da (CUA)', 0.10, 1.00, 0.05),
    ('control.pid.kp', 'PID Kp', 0.0, 2.0, 0.05),
    ('control.pid.kd', 'PID Kd', 0.0, 1.0, 0.01),
    ('control.steer_lookahead_weight', 'Trong so diem ngam', 0.0, 1.0, 0.05),
    ('control.slowdown', 'Bo ga theo lech', 0.0, 1.0, 0.01),
    ('control.v_min', 'Ga toi thieu', 0.00, 0.40, 0.01),
    ('control.v_max', 'Tran ga (an toan)', 0.00, 0.80, 0.01),
]


# Slider rieng cho che do LAI TAY. Tach khoi CONTROL_PARAMS vi day la gioi han
# cua NGUOI lai, khong lien quan gi den bo dieu khien tu dong - tron hai nhom se
# lam nguoi dung chinh nham nhom khi dang lai tay.
MANUAL_PARAMS = [
    ('manual.max_steering', 'Bo goc toi da', 0.10, 1.00, 0.05),
    ('manual.steering_expo', 'Do mem lai (expo)', 0.00, 0.90, 0.05),
    ('manual.steering_slew_rate', 'Toc do doi lai', 0.5, 8.0, 0.1),
    ('manual.max_throttle', 'Toc do toi da', 0.05, 0.60, 0.01),
    ('manual.min_throttle', 'Ga khoi dong', 0.00, 0.40, 0.01),
    ('manual.throttle_rise_rate', 'Toc do len ga', 0.2, 5.0, 0.1),
    ('manual.throttle_fall_rate', 'Toc do nha ga', 0.5, 10.0, 0.5),
    ('manual.deadzone', 'Vung chet can', 0.00, 0.30, 0.01),
]


class RollingStats(object):
    """Thong ke cua so truot - de bat loi, khong phai de bao cao ket qua.

    So lieu chinh thuc cho paper phai lay tu file CSV cua mot luot chay day du,
    khong phai tu cua so truot cua giao dien tune.
    """

    def __init__(self, window=200):
        self.window = int(window)
        self._lock = threading.Lock()
        self._cte = []
        self._steer = []
        self._found = []
        self._bands = []

    def push(self, cte, steer, found, bands):
        with self._lock:
            for buf, value in ((self._cte, cte), (self._steer, steer),
                               (self._found, 1.0 if found else 0.0),
                               (self._bands, bands)):
                buf.append(float(value))
                if len(buf) > self.window:
                    buf.pop(0)

    def reset(self):
        with self._lock:
            del self._cte[:], self._steer[:], self._found[:], self._bands[:]

    @property
    def n(self):
        with self._lock:
            return len(self._cte)

    def summary(self):
        with self._lock:
            if not self._cte:
                return None
            cte = np.array(self._cte)
            steer = np.array(self._steer)
            found = np.array(self._found)
            bands = np.array(self._bands)
        sign = np.sign(steer)
        flips = int(np.sum(sign[1:] * sign[:-1] < 0)) if len(steer) > 1 else 0
        return {
            'n': len(cte),
            'cte_rms': float(np.sqrt(np.mean(cte ** 2))),
            'cte_p95': float(np.percentile(np.abs(cte), 95)),
            'loss_pct': float(100.0 * (1.0 - np.mean(found))),
            'bands_mean': float(np.mean(bands)),
            'steer_flips': flips,
            'steer_sat_pct': float(100.0 * np.mean(np.abs(steer) >= 0.999 * (
                np.max(np.abs(steer)) if np.max(np.abs(steer)) > 0 else 1.0))),
        }

    def cte_history(self):
        with self._lock:
            return list(self._cte)

    def steer_history(self):
        with self._lock:
            return list(self._steer)


# Ba che do lai. Tach ro de khong bao gio co chuyen "tuong dang tay ma xe tu chay".
MODE_STOP = 'DUNG'
MODE_MANUAL = 'TAY CAM'
MODE_AUTO = 'TU DONG'


class ProcessingFpsMeter(object):
    """Do nhip HOAN TAT frame cua pipeline, khong lien quan nhip ve UI.

    Dung khoang cach giua cac moc hoan tat thay vi ``1 / process_ms``. Nhu vay
    thoi gian cho camera/frame moi va cac buoc ghi log nhe van duoc tinh, con
    render panel/JPEG (chay o worker rieng) khong lam sai con so.
    """

    def __init__(self, window=60):
        self.window = max(2, int(window))
        self.reset()

    def reset(self):
        self._intervals = deque(maxlen=self.window)
        self._first = None
        self._last = None
        self.frames = 0

    def tick(self, completed_at=None):
        now = time.monotonic() if completed_at is None else float(completed_at)
        if self._last is not None:
            dt = now - self._last
            if dt > 0.0:
                self._intervals.append(dt)
        else:
            self._first = now
        self._last = now
        self.frames += 1
        return self.fps

    @property
    def fps(self):
        if not self._intervals:
            return 0.0
        mean_dt = sum(self._intervals) / float(len(self._intervals))
        return 1.0 / mean_dt if mean_dt > 0.0 else 0.0

    def current_fps(self, now=None, stale_after=None):
        """FPS rolling, ve 0 neu qua lau khong co completion moi."""
        if self._last is None:
            return 0.0
        current = time.monotonic() if now is None else float(now)
        if (stale_after is not None and
                current - self._last > float(stale_after)):
            return 0.0
        return self.fps

    @property
    def mean_fps(self):
        if self.frames < 2 or self._first is None or self._last is None:
            return 0.0
        elapsed = self._last - self._first
        return ((self.frames - 1) / elapsed) if elapsed > 0.0 else 0.0


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
        # Nguon da tu sua mau (camera.shading.apply_at = source) thi o day
        # KHONG duoc sua nua - sua hai lan se day gain len binh phuong, anh xam
        # thanh xanh la va moi nguong mau da tune deu sai.
        if shading_applied_at_source(cfg):
            self.shading = ShadingCorrector.disabled()
            self.shading_at_source = True
        else:
            self.shading = ShadingCorrector.from_config(cfg)
            self.shading_at_source = False
        # Giu lai he so dang co hieu luc (du sua o dau) de ghi vao metadata.
        _effective = ShadingCorrector.from_config(cfg)
        self.shading_coeff_r = list(_effective.coeff_r)
        self.shading_coeff_b = list(_effective.coeff_b)
        prev = getattr(self, 'lane', None)
        # Detector CNN giu mot engine TensorRT nang. Chi doi tham so, khong
        # dung lai - xem CnnLaneDetector.update_config().
        if (prev is not None and hasattr(prev, 'update_config')
                and getattr(prev, 'engine_path', None)
                == cfg.get('lane.cnn.engine', '')):
            prev.update_config(cfg)
        else:
            self.lane = build_lane_detector(cfg)
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
        # Tham so cua driver, chi de HIEN THI output servo that - vong tune
        # khong tu doi hard-limit cua driver (phai hieu chuan bang
        # tools/check_hardware.py --calibrate-steering, xe ke banh).
        self.drv_gain = float(cfg.get('control.driver.steering_gain', -0.65))
        self.drv_offset = float(cfg.get('control.driver.steering_offset', 0.0))
        self.drv_out_min = float(
            cfg.get('control.driver.steering_output_min', -1.0))
        self.drv_out_max = float(
            cfg.get('control.driver.steering_output_max', 1.0))
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

        servo, servo_raw = servo_output_for(
            steer, self.drv_gain, self.drv_offset,
            self.drv_out_min, self.drv_out_max)
        # Hai cho co the cat lenh lai, phai phan biet duoc:
        #   - `steer_clipped`: bi tran phan mem (steer_max / corner_steer_max)
        #   - `servo_clipped`: bi hard-limit cua driver (steering_output_max)
        # Chi bao "lai toi da" chung chung thi nguoi dung se chinh nham num.
        steer_clipped = abs(self.corner.steer_wanted) > self.corner.steer_limit + 1e-6
        servo_clipped = abs(servo_raw - servo) > 1e-6

        self.stats.push(res.cte, steer, res.found, res.n_bands)
        result = {
            'proc': proc, 'lane': res, 'steer': steer, 'throttle': throttle,
            'error': err, 'ramp': ramp, 'mode': mode, 'pid': pid_out,
            'servo': servo, 'servo_span': self.drv_out_max,
            'steer_clipped': steer_clipped, 'servo_clipped': servo_clipped,
            'steer_wanted': self.corner.steer_wanted,
        }
        lane_mask = getattr(self.lane, 'last_mask', None)
        if lane_mask is not None:
            # Chi ~32 KiB voi mask CNN 256x128; doi lai preview khong can khoa
            # TensorRT detector va khong the lay nham mask cua frame sau.
            result['_preview_mask'] = lane_mask.copy()
        return result

    # ------------------------------------------------------------------ render
    def render_panel(self, result, fps=0.0, armed=False, width=None,
                     driver_kind='dryrun', ui_mode=None, override_cmd=None):
        """Panel 2x2: anh+mask, mask nhi phan, bird's-eye+fit, do thi CTE."""
        res = result['lane']
        proc = result['proc']
        ph, pw = proc.shape[:2]

        y0 = int(ph * float(self.cfg.get('lane.roi_top', 0.55)))
        mask = np.zeros((ph, pw), np.uint8)
        lane_mask = result.get('_preview_mask')
        if hasattr(self.lane, '_binarise'):
            # CV co dien: mask mau tinh lai tren dung anh proc dang hien thi.
            if ph > y0:
                mask[y0:, :] = self.lane._binarise(proc[y0:, :])
        elif lane_mask is not None or getattr(self.lane, 'last_mask', None) is not None:
            # CNN: mask model xuat ra o khong gian ROI (256x128) - keo ve dung o
            # ROI cua anh proc de chong len camera. KHONG dung `res.debug`: cho
            # do la anh BEV, dung cho panel 3.
            if ph > y0:
                # Pipeline va preview chay hai thread rieng. Snapshot nay duoc
                # chup ngay sau inference, tranh preview doc ``last_mask`` cua
                # frame ke tiep trong khi dang ve frame hien tai.
                if lane_mask is None:
                    lane_mask = self.lane.last_mask
                mask[y0:, :] = cv2.resize(lane_mask, (pw, ph - y0),
                                          interpolation=cv2.INTER_NEAREST)

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
        if override_cmd is not None:
            # Dang LAI TAY: banner phai hien lenh NGUOI dang gui, khong phai
            # lenh CV. Hien lenh CV luc do la sai su that - nguoi dung se tuong
            # minh dang be lai 0.6 trong khi thuc te tay cam gui so khac.
            man_steer, man_throttle = override_cmd
            servo, servo_raw = servo_output_for(
                man_steer, self.drv_gain, self.drv_offset,
                self.drv_out_min, self.drv_out_max)
            result = dict(result)
            result['steer'] = man_steer
            result['throttle'] = man_throttle
            result['servo'] = servo
            result['steer_clipped'] = False
            result['servo_clipped'] = abs(servo_raw - servo) > 1e-6
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

    Hai khoi chu duoc do be rong THAT roi moi dat cho: khoi trai canh trai,
    khoi phai canh phai. Dat o ty le co dinh thi chuoi dai ngan khac nhau tuy
    che do se de len nhau va khong doc duoc con so nao.
    """
    width = panel.shape[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    banner = np.zeros((58, width, 3), np.uint8)
    banner[:] = (18, 18, 18)

    live = (str(driver_kind) != 'dryrun')
    ramp = float(result.get('ramp', 1.0))
    if armed and ui_mode == MODE_MANUAL:
        # O che do nay CV van chay va van hien so lieu, nhung NGUOI moi la
        # nguoi dieu khien xe - phai phan biet ro.
        state = ('LAI TAY - ban dieu khien' if live
                 else 'LAI TAY (DRYRUN - banh khong quay)')
        colour = (255, 160, 60)
    elif not armed:
        state, colour = 'DANG DUNG (bam CHAY hoac LAI TAY)', (160, 160, 160)
    elif not live:
        state, colour = 'DRYRUN - BANH KHONG QUAY', (0, 200, 255)
    elif ramp < 1.0:
        state, colour = 'DANG CHAY - tang ga %d%%' % int(ramp * 100), (0, 200, 255)
    else:
        state, colour = 'DANG CHAY - BAM LINE', (60, 60, 255)

    mode = result.get('mode') or ''
    found = 'CO' if res.found else 'MAT VACH'
    left = [
        (state, 0.5, colour, 2),
        ('vach=%s  dai=%d  FPS(XU LY)=%.1f  %s' %
         (found, res.n_bands, fps, mode),
         0.40, (120, 240, 120) if res.found else (60, 60, 255), 1),
    ]

    servo = result.get('servo')
    if servo is None:
        steer_line = 'lai=%+.3f  ga=%.3f' % (result['steer'], result['throttle'])
        steer_colour = (240, 240, 240)
    else:
        if result.get('servo_clipped'):
            tag, steer_colour = ' HARD-LIMIT', (60, 60, 255)
        elif result.get('steer_clipped'):
            tag, steer_colour = ' TRAN LAI', (0, 200, 255)
        else:
            tag, steer_colour = '', (240, 240, 240)
        steer_line = 'lai%+.2f servo%+.2f%s ga%.2f' % (
            result['steer'], servo, tag, result['throttle'])
    right = [
        ('cte%+.2f ngam%+.2f cong%+.2f' % (
            res.cte, res.cte_lookahead, res.curvature), 0.42, (240, 240, 240)),
        (steer_line, 0.42, steer_colour),
    ]

    left_w = max(cv2.getTextSize(t, font, sc, th)[0][0] for t, sc, _c, th in left)
    right_w = max(cv2.getTextSize(t, font, sc, 1)[0][0] for t, sc, _c in right)
    x_right = max(left_w + 16, width - right_w - 8)

    for i, (text, scale, col, thick) in enumerate(left):
        cv2.putText(banner, text, (8, 20 + i * 24), font, scale, col, thick)
    for i, (text, scale, col) in enumerate(right):
        cv2.putText(banner, text, (x_right, 20 + i * 24), font, scale, col, 1)
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
                 save_path='configs/tuned.yaml', preview_width=640,
                 soft_start_s=1.0, record_dir='logs', record_fps=15.0,
                 controller_index=0, data_root='data/driving',
                 control_hz=30.0, overrides=None, simple=False):
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

        self.simple = bool(simple)
        self.widgets = widgets
        self.source_kind = source_kind
        self.video_path = video_path
        self.driver_kind = driver_kind
        self.save_path = save_path
        self.preview_width = preview_width

        self.engine = LaneTuningEngine(config_path, overrides)
        self.engine.soft_start_s = float(soft_start_s)
        self.shaper = ControllerShaper(self.engine.cfg)
        self.controller = widgets.Controller(index=int(controller_index))
        self.data_root = data_root
        self.control_period = 1.0 / max(5.0, float(control_hz))

        self._grabber = None
        self._driver = None
        self._armed = False
        self._stop_event = threading.Event()
        self._camera_stopping = threading.Event()
        self._thread = None
        self._driver_lock = threading.Lock()
        # Bao ve chuyen trang thai camera/ARM/log giua callback Jupyter va
        # thread camera. RLock cho phep _disarm() duoc goi tu transition dang
        # giu khoa ma khong tu deadlock.
        self._lifecycle_lock = threading.RLock()
        self._engine_lock = threading.RLock()
        # FPS nay la nhip frame HOAN TAT pipeline camera -> CV -> lenh. Preview
        # va JPEG chay o thread khac, nen khong con keo con so nay xuong.
        self._fps = 0.0
        self._pipeline_meter = ProcessingFpsMeter(
            self.engine.get_param('pipeline.fps_window', 60))
        self._run_meter = ProcessingFpsMeter(
            self.engine.get_param('pipeline.fps_window', 60))
        camera_fps = float(self.engine.get_param('camera.fps', 30.0))
        default_stall_s = max(0.25, 2.5 / max(1.0, camera_fps))
        self._fps_stale_after = max(
            0.10, float(self.engine.get_param(
                'tuning.pipeline_stall_s', default_stall_s)))
        self.record_dir = record_dir
        self.record_fps = float(record_fps)
        self._recorder = None
        self._record_lock = threading.Lock()
        # Vong driver chay 30 Hz doc lap: inference co the dao dong, con preview
        # da o worker rieng. Driver chi doc lenh CV moi nhat, khong cho jitter
        # cua perception bien thanh jitter servo/watchdog.
        self._control_thread = None
        self._control_stop = threading.Event()
        self._state_lock = threading.Lock()
        self._cv_steer = 0.0
        self._cv_throttle = 0.0
        self._cv_started_mono = None
        self._man_steer = 0.0
        self._man_throttle = 0.0
        self._stick_raw = (0.0, 0.0)
        self._deadman_ok = True
        self.mode = MODE_STOP
        self._writer = None
        self._writer_lock = threading.Lock()
        self._last_sample = 0.0

        # Render panel + encode JPEG la viec cua UI, khong thuoc pipeline dieu
        # khien. Worker chi giu payload moi nhat (khong xep hang frame cu).
        preview_hz = float(self.engine.get_param('tuning.preview_hz', 5.0))
        self._preview_period = 1.0 / max(1.0, preview_hz)
        self._preview_lock = threading.Lock()
        self._preview_event = threading.Event()
        self._preview_stop = threading.Event()
        self._preview_payload = None
        self._preview_thread = None

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
        # Chon dung bo slider theo nguon bam vach dang chay.
        self.dung_cnn = type(self.engine.lane).__name__ == 'CnnLaneDetector'
        self.lane_box = self._build_sliders(
            CNN_LANE_PARAMS if self.dung_cnn else LANE_PARAMS)
        self.simple_box = None      # che do don gian tu dung num rieng
        self._logger = None
        self._logger_lock = threading.Lock()
        self._run_mode = MODE_STOP
        self._run_started_mono = None
        self._last_run_summary = None
        self.log_dir = str(self.engine.get_param('logging.dir', 'logs'))
        self.control_box = self._build_sliders(CONTROL_PARAMS)
        self.manual_box = self._build_sliders(MANUAL_PARAMS,
                                              on_change=self._apply_manual)

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
        # Ghi CSV ngay tu giao dien. Truoc day chi CLI moi ghi, nen muon co so
        # lieu de phan tich la phai bo giao dien ra chay lenh khac - it ai lam,
        # va the la tune bang cam giac.
        self.btn_log = widgets.Button(description='LOG: tu dong khi CHAY',
                                      button_style='info', disabled=True,
                                      tooltip='Moi luot CHAY/LAI TAY tao mot CSV; '
                                              'DUNG se flush va dong file.')
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
    def _apply_manual(self):
        """Ap tham so tay cam ngay lap tuc.

        KHONG goi shaper.reset(): reset se dua lenh ve 0 va xe khu ga giua chung
        moi lan keo mot slider. `reset_from_config` chi doi gioi han, giu nguyen
        lenh dang chay.
        """
        with self._lifecycle_lock:
            with self._engine_lock:
                self.shaper.reset_from_config(self.engine.cfg)

    def _build_sliders(self, params, on_change=None):
        w = self.widgets
        rows = []
        for row in params:
            key, label, lo, hi, step = row[:5]
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
            slider.observe(self._make_setter(key, on_change), names='value')
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
        if key.startswith('manual.'):
            # Lay mac dinh THAT cua ControllerShaper, khong lay giua thang do:
            # giua thang do cua `deadzone` la 0.15 - qua lon, can gat gan nhu
            # khong an.
            attr = key.split('.', 1)[1]
            value = getattr(self.shaper, attr, None)
            if value is not None:
                return value
        return (lo + hi) / 2.0

    def _make_setter(self, key, on_change=None):
        def handler(change):
            with self._engine_lock:
                self.engine.set_param(key, change['new'])
            if on_change is not None:
                on_change()
        return handler

    def _on_colour_change(self, change):
        with self._engine_lock:
            self.engine.set_param('lane.line_color', change['new'])
            # Nguong S/V phu thuoc mau vach -> keo slider ve mac dinh cua preset
            # moi; RLock cho phep observer cua slider vao lai cung transaction.
            for key in ('lane.hsv_s_min', 'lane.hsv_v_min',
                        'lane.min_blob_area'):
                if key in self._sliders:
                    default = self._default_for(key, 0, 255)
                    self._sliders[key].value = default
                    self.engine.set_param(key, default)
        self._log('Doi mau vach sang "%s"; nguong S/V/blob da ve mac dinh preset.'
                  % change['new'])

    def _on_shading_change(self, change):
        try:
            with self._engine_lock:
                self.engine.set_param(
                    'camera.shading.enabled', bool(change['new']))
                self.engine.rebuild()
        except IOError as exc:
            self.use_shading.value = False
            with self._engine_lock:
                self.engine.set_param('camera.shading.enabled', False)
            self._log('Khong bat duoc sua mau: %s' % exc)

    def _log(self, message):
        with self.output:
            print('[%s] %s' % (time.strftime('%H:%M:%S'), message))

    def _set_status(self, message):
        self.status.value = '<b>Trang thai:</b> %s' % message

    # -------------------------------------------------------------------- loop
    def _loop(self, grabber):
        stop_reason = 'camera_stop'
        try:
            stop_reason = self._process_loop(grabber)
        except Exception as exc:
            stop_reason = 'pipeline_error'
            try:
                self._log('LOI PIPELINE -> DISARM: %s' % exc)
                self.status.value = (
                    '<div style="padding:8px;background:#b00020;color:#fff;'
                    'font-weight:bold">PIPELINE DA DUNG DO LOI: %s</div>' % exc)
            except Exception:
                pass
        finally:
            self._finalize_camera_loop(grabber, stop_reason)

    def _process_loop(self, grabber):
        last_id = -1
        last_preview = 0.0
        last_result = None
        t_prev = None
        stop_reason = 'camera_stop'

        while not self._stop_event.is_set():
            pipeline_t0 = time.monotonic()
            frame, frame_id = grabber.read()
            if frame is None or frame_id == last_id:
                fresh_fps = self._pipeline_meter.current_fps(
                    time.monotonic(), self._fps_stale_after)
                if self._fps > 0.0 and fresh_fps == 0.0:
                    self._fps = 0.0
                    if last_result is not None:
                        self._queue_preview(last_result)
                if getattr(grabber, 'eof', False) or grabber.error is not None:
                    self._camera_stopping.set()
                    if grabber.error is not None:
                        self._log('CAMERA DUNG DO LOI: %s' % grabber.error)
                        stop_reason = 'camera_error'
                    else:
                        self._log('Nguon camera/video da het.')
                        stop_reason = 'camera_eof'
                    break
                time.sleep(0.005)
                continue
            last_id = frame_id

            now = time.time()
            now_mono = time.monotonic()
            dt = (1.0 / 30.0) if t_prev is None else max(1e-6, now_mono - t_prev)
            t_prev = now_mono

            try:
                with self._engine_lock:
                    result = self.engine.process(frame, dt, now=now)
            except Exception as exc:
                # Bao THAT TO. Truoc day chi ghi mot dong log duoi cung: vong
                # lap break, panel dung im o khung hinh cuoi, va nguoi dung
                # tuong model "detect ra mot cuc do roi khong doi" chu khong
                # biet la da co loi. Mat ca buoi de tim ra.
                self._log('LOI XU LY -> DISARM: %s' % exc)
                self.status.value = (
                    '<div style="padding:8px;border-radius:4px;background:#b00020;'
                    'color:#fff;font-weight:bold">DA DUNG VI LOI XU LY - '
                     'panel ben duoi la KHUNG HINH CU, khong phai ket qua moi.'
                     '<br>%s</div>' % exc)
                stop_reason = 'processing_error'
                self._camera_stopping.set()
                break

            # Cong bo lenh CV la bien CUOI cua pipeline can do. Render/JPEG
            # khong nam trong do va duoc day sang preview worker phia duoi.
            with self._state_lock:
                self._cv_steer = result['steer']
                self._cv_throttle = result['throttle']
                self._cv_started_mono = pipeline_t0

            completed = time.monotonic()
            pipeline_latency_ms = (completed - pipeline_t0) * 1000.0
            self._fps = self._pipeline_meter.tick(completed)
            last_result = result
            self._write_log_row(
                result, frame_id, now, completed, pipeline_latency_ms,
                pipeline_started_mono=pipeline_t0)
            self._record_frame(frame, frame_id, result, now)
            self._record_dataset(frame, frame_id, result, now,
                                 completed)

            if (completed - last_preview) >= self._preview_period:
                last_preview = completed
                self._queue_preview(result)

        return stop_reason

    def _finalize_camera_loop(self, grabber, stop_reason):
        self._camera_stopping.set()
        # Camera dung va DISARM la mot transition. Neu bam CHAY dung luc camera
        # loi, khoa nay ngan logger/ARM "song lai" sau khi thread da dung.
        try:
            with self._lifecycle_lock:
                if self._grabber is grabber:
                    self._grabber = None
                self._disarm_locked(stop_reason)
        except Exception as exc:
            try:
                self._log('Loi DISARM khi dong camera: %s' % exc)
            except Exception:
                pass
        try:
            with self._record_lock:
                if self._recorder is not None:
                    self._log('Camera dung -> tu dong dong file ghi.')
                    self._stop_record_locked()
        except Exception as exc:
            try:
                self._log('Loi dong file ghi camera: %s' % exc)
            except Exception:
                pass
        try:
            with self._writer_lock:
                if self._writer is not None:
                    self._log('Camera dung -> tu dong dong session data.')
                    self._stop_data_locked()
        except Exception as exc:
            try:
                self._log('Loi dong session data: %s' % exc)
            except Exception:
                pass
        try:
            grabber.stop()
        except Exception:
            pass
        try:
            self._set_status('camera da dung; bam MO CAMERA de mo lai')
        except Exception:
            pass

    def _manual_command(self):
        with self._state_lock:
            return (self._man_steer, self._man_throttle)

    def _ensure_preview_thread(self):
        if self._preview_thread is not None and self._preview_thread.is_alive():
            return
        self._preview_stop.clear()
        self._preview_thread = threading.Thread(target=self._preview_loop)
        self._preview_thread.daemon = True
        self._preview_thread.start()

    def _queue_preview(self, result):
        payload = (result, self._fps, self._armed, self.mode,
                   self._manual_command() if self.mode == MODE_MANUAL else None)
        with self._preview_lock:
            # Slot latest-only: UI cham thi bo panel cu, khong bao gio lam tre
            # lenh dieu khien hoac tich mot queue anh ton RAM.
            self._preview_payload = payload
        self._preview_event.set()

    def _preview_loop(self):
        while not self._preview_stop.is_set():
            self._preview_event.wait(0.25)
            if self._preview_stop.is_set():
                break
            self._preview_event.clear()
            with self._preview_lock:
                payload = self._preview_payload
                self._preview_payload = None
            if payload is None:
                continue
            result, fps, armed, mode, manual_cmd = payload
            try:
                panel = self.engine.render_panel(
                    result, fps=fps, armed=armed, width=self.preview_width,
                    driver_kind=self.driver_kind, ui_mode=mode,
                    override_cmd=manual_cmd)
                data = self.engine.encode_jpeg(panel)
                if data:
                    self.preview.value = data
                self.metrics.value = self._metrics_html()
            except Exception as exc:
                self._log('Loi ve preview: %s' % exc)

    # ----------------------------------------------------------- log tung luot
    def _start_run_log(self, mode):
        from .logging_csv import RunLogger
        if self._logger is not None:
            self._finish_run_log('restarted')
        task = 'tune_auto' if mode == MODE_AUTO else 'tune_manual'
        logger = None
        try:
            logger = RunLogger(self.log_dir, task)
            # Khong tinh thoi gian mo/tao file tren SD vao FPS cua luot chay.
            started_mono = time.monotonic()
            started_wall = time.time()
            logger.mark_start(started_wall)
            logger.write(timestamp=started_wall, frame_id='', decision='start',
                         state='STARTING', event='start')
        except Exception as exc:
            if logger is not None:
                try:
                    logger.close()
                except Exception:
                    pass
            self._log('KHONG TAO DUOC LOG -> TU CHOI CHAY: %s' % exc)
            return False
        with self._logger_lock:
            if self._logger is not None:
                # Phong thu: binh thuong moi luot da duoc close khi DUNG.
                try:
                    self._logger.close()
                except Exception:
                    pass
            self._logger = logger
            self._run_mode = mode
            self._run_meter.reset()
            self._run_started_mono = started_mono
            self._last_run_summary = None
        self.btn_log.description = 'DANG TU GHI LOG'
        self.btn_log.button_style = 'warning'
        self._log('Tu dong bat dau log %s: %s' % (mode, logger.path))
        return True

    def _finish_run_log(self, event='stop'):
        message = None
        with self._logger_lock:
            logger = self._logger
            if logger is None:
                return None
            self._logger = None
            stopped_mono = time.monotonic()
            duration = (0.0 if self._run_started_mono is None else
                        max(0.0, stopped_mono - self._run_started_mono))
            fps_mean = (self._run_meter.frames / duration
                        if duration > 0.0 else 0.0)
            fps_window = self._run_meter.current_fps(
                stopped_mono, self._fps_stale_after)
            frames = self._run_meter.frames
            close_error = None
            try:
                logger.write(
                    timestamp=time.time(), frame_id=frames,
                    fps=fps_window, decision='stop', state='FINISHED',
                    event=event)
            except Exception as exc:
                close_error = exc
            finally:
                try:
                    logger.close()
                except Exception as exc:
                    if close_error is None:
                        close_error = exc
            summary = {
                'path': logger.path, 'frames': frames,
                'fps_mean': fps_mean, 'fps_window': fps_window,
                'seconds': duration, 'event': event, 'mode': self._run_mode,
            }
            self._last_run_summary = summary
            self._run_mode = MODE_STOP
            self._run_started_mono = None
            message = ('Da dong log: %s (%d frame, FPS xu ly TB %.2f, '
                       'FPS cua so %.2f, ly do=%s)'
                       % (logger.path, frames, fps_mean, fps_window, event))
            if close_error is not None:
                message += ' [LOI KHI CHOT LOG: %s]' % close_error
        self.btn_log.description = 'LOG: tu dong khi CHAY'
        self.btn_log.button_style = 'info'
        self._log(message)
        return summary

    def _on_log(self, _b=None):
        """Giu API cu, nhung log bay gio gan bat buoc voi mot luot chay."""
        self._log('Log la tu dong: bam CHAY/LAI TAY de mo, bam DUNG de flush.')

    def _write_log_row(self, result, frame_id, now, completed,
                       pipeline_latency_ms, pipeline_started_mono=None):
        res = result['lane']
        error = None
        # Doi transition CHAY hoan tat. Frame bat dau luc UI con DUNG khong
        # thuoc luot moi, du no hoan tat sau khi logger vua duoc mo.
        with self._lifecycle_lock:
            if not self._armed:
                return
            run_mode = self._run_mode
            if run_mode == MODE_MANUAL:
                with self._state_lock:
                    log_steer = self._man_steer
                    log_throttle = self._man_throttle
                log_decision = 'MANUAL'
                log_drive_mode = 'MANUAL'
            else:
                log_steer = result['steer']
                log_throttle = result['throttle']
                log_decision = result['mode']
                log_drive_mode = result['mode']
        with self._logger_lock:
            logger = self._logger
            if logger is None:
                return
            if (pipeline_started_mono is not None and
                    self._run_started_mono is not None and
                    pipeline_started_mono < self._run_started_mono):
                return
            run_fps = self._run_meter.tick(completed)
            event = ('HARD_LIMIT'
                     if (run_mode != MODE_MANUAL and
                         result.get('servo_clipped')) else '')
            try:
                logger.write(
                    timestamp=now, frame_id=frame_id, fps=run_fps,
                    latency_ms=pipeline_latency_ms,
                    decision=log_decision,
                    control_output='steer=%.3f;throttle=%.3f'
                                   % (log_steer, log_throttle),
                    cte=res.cte, cte_lookahead=res.cte_lookahead,
                    curvature=res.curvature, lane_found=1 if res.found else 0,
                    n_bands=res.n_bands, throttle=log_throttle,
                    drive_mode=log_drive_mode,
                    state='RUNNING', event=event)
            except Exception as exc:
                error = exc
        if error is not None:
            # Chi dung neu logger vua loi van la logger hien tai. Khong de loi
            # tre cua luot A dong nham luot B vua duoc nguoi dung khoi dong.
            with self._lifecycle_lock:
                if self._logger is logger:
                    self._log('Loi ghi log -> dung luot va chot file: %s'
                              % error)
                    self._disarm_locked('logging_error')

    def _sync_driver_limits(self):
        """Day tran servo tu config xuong doi tuong driver.

        Driver clip LAN THU HAI o gia tri no nhan luc duoc tao, va no khong tu
        doc lai config. Khong dong bo thi keo slider TRAN SERVO se doi con so o
        engine ma driver van cat o muc cu -> slider nhin nhu hong.
        """
        drv = self._driver
        if drv is None:
            return
        for attr, key, dflt in (
                ('steering_output_min', 'control.driver.steering_output_min', -1.0),
                ('steering_output_max', 'control.driver.steering_output_max', 1.0)):
            if hasattr(drv, attr):
                val = float(self.engine.get_param(key, dflt))
                if abs(getattr(drv, attr) - val) > 1e-9:
                    setattr(drv, attr, val)

    def _metrics_html(self):
        s = self.engine.stats.summary()
        if s is None:
            return '<i>chua co so lieu</i>'
        warn = []
        span = abs(self.engine.drv_out_max)
        reach = min(abs(self.engine.corner.corner_steer_max
                        * self.engine.drv_gain), span)
        if reach < 0.95 * 1.0:
            warn.append(
                'servo chi dung %.0f%% tam quay (lai %.2f x gain %.2f, '
                'hard-limit %.2f) -> ban kinh cua bi rong. Hieu chuan bang '
                'tools/check_hardware.py --calibrate-steering'
                % (100 * reach, self.engine.corner.corner_steer_max,
                   abs(self.engine.drv_gain), span))
        if s['loss_pct'] > 2.0:
            warn.append('mat vach %.1f%% (muc tieu <= 2%%)' % s['loss_pct'])
        if s['bands_mean'] < 3.0:
            warn.append('chi %.1f dai - it du lieu, giam "Dien tich blob"'
                        % s['bands_mean'])
        if s['cte_rms'] > 0.15:
            warn.append('cte_rms %.3f (muc tieu <= 0.15)' % s['cte_rms'])
        if self._fps > 0.0 and self._fps < 20.0:
            warn.append('FPS xu ly %.1f < 20 (khong tinh render/JPEG)' % self._fps)
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
            '<tr><td>FPS xu ly</td><td><b>%.1f</b></td>'
            '<td style="padding-left:18px">muc tieu</td><td><b>&gt;=20</b></td></tr>'
            '</table>' % (banner, s['n'], s['loss_pct'], s['cte_rms'],
                          s['cte_p95'], s['bands_mean'], s['steer_flips'],
                          self._fps))

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

        Preview/JPEG da tach rieng, nhung inference van co the dao dong theo
        frame. Driver giu nhip 30 Hz doc lap de tay cam va watchdog khong giat.
        """
        t_prev = time.monotonic()
        while not self._control_stop.is_set():
            t0 = time.monotonic()
            dt = t0 - t_prev
            t_prev = t0

            driver_error = None
            camera_stall = False
            # Giu lifecycle qua ca luc doc mode va gui lenh. Nhu vay DUNG khong
            # the chen giua phep check `_armed` va driver.set(), roi bi mot lenh
            # stale bat motor lai sau driver.stop(). Thu tu khoa luon la
            # lifecycle -> driver, giong cac transition CHAY/DUNG.
            with self._lifecycle_lock:
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
                        cv_started_mono = self._cv_started_mono
                    # Lenh sinh tu frame bat dau truoc moc CHAY la lenh cu.
                    # Gui neutral den khi co completion dau tien cua luot moi.
                    if (self._run_started_mono is None or
                            cv_started_mono is None or
                            cv_started_mono < self._run_started_mono):
                        steer, throttle = 0.0, 0.0
                        if (self._run_started_mono is not None and
                                t0 - self._run_started_mono >
                                self._fps_stale_after):
                            camera_stall = True
                    elif t0 - cv_started_mono > self._fps_stale_after:
                        steer, throttle = 0.0, 0.0
                        camera_stall = True
                else:
                    steer, throttle = 0.0, 0.0
                    self.shaper.reset()

                if camera_stall and self._armed:
                    self._camera_stopping.set()
                    self._stop_event.set()
                    self._fps = 0.0
                    self._log('PIPELINE MAT FRAME -> dung xe (command qua %.2fs)'
                              % self._fps_stale_after)
                    self._disarm_locked('camera_stall')
                    self._set_status('DA DUNG - pipeline mat frame, motor=0')
                    self.metrics.value = self._metrics_html()
                elif self._armed and self._driver is not None:
                    with self._driver_lock:
                        try:
                            self._sync_driver_limits()
                            self._driver.set(steer, throttle)
                        except Exception as exc:
                            driver_error = exc
                # Da thoat driver_lock nhung van giu lifecycle: loi cua luot cu
                # khong the chen vao va DISARM mot luot moi.
                if driver_error is not None:
                    self._log('LOI DRIVER -> dung xe: %s' % driver_error)
                    self._disarm_locked('driver_error')

            sleep = self.control_period - (time.monotonic() - t0)
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
            # Truy nguoc duoc anh nay da bi sua mau bang he so nao. Khi
            # sua tai nguon thi khong the do nguoc lai tu anh nua.
            'shading_enabled': bool(cfg.get('camera.shading.enabled', False)),
            'shading_apply_at': cfg.get('camera.shading.apply_at', 'source'),
            'shading_coeff_r': list(self.engine.shading_coeff_r),
            'shading_coeff_b': list(self.engine.shading_coeff_b),
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
        with self._lifecycle_lock:
            return self._enter_mode_locked(mode)

    def _enter_mode_locked(self, mode):
        """Doi che do lai. Tra ve True neu vao duoc."""
        if (self._grabber is None or self._camera_stopping.is_set() or
                getattr(self._grabber, 'eof', False) or
                getattr(self._grabber, 'error', None) is not None):
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

        try:
            self.shaper.reset_from_config(self.engine.cfg)
            self.shaper.reset()
            self.mode = mode
            if mode == MODE_MANUAL:
                with self._engine_lock:
                    self.engine.stop_run()
            if not self._start_run_log(mode):
                self.mode = MODE_STOP
                return False
            self._ensure_control_thread()
            self._armed = True
            return True
        except Exception as exc:
            self._armed = False
            self.mode = MODE_STOP
            with self._engine_lock:
                self.engine.stop_run()
            self._finish_run_log('start_error')
            self._set_status('LOI KHOI DONG - xe khong chay')
            self._log('Khong vao duoc che do %s: %s' % (mode, exc))
            return False

    def _on_manual(self, _b=None):
        if self.mode == MODE_MANUAL:
            self._on_disarm()
            return
        if self._armed:
            self._log('TU CHOI LAI TAY: xe dang TU DONG. Bam DUNG truoc khi '
                      'chuyen che do.')
            self._set_status('Dang TU DONG - DUNG truoc khi chuyen LAI TAY')
            return
        if not self._enter_mode(MODE_MANUAL):
            return
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
        with self._lifecycle_lock:
            return self._on_open_locked(_b)

    def _on_open_locked(self, _b=None):
        if self._grabber is not None:
            self._log('Camera dang mo.')
            return
        source = None
        grabber = None
        try:
            self._log('Dang mo camera kind=%s, %s'
                      % (self.source_kind, format_camera_environment()))
            source = build_source(self.engine.cfg, self.source_kind,
                                  video_path=self.video_path)
            grabber = LatestFrameGrabber(source).start()
            if grabber.error is not None:
                raise RuntimeError(grabber.error)
        except Exception as exc:
            if grabber is not None:
                try:
                    grabber.stop()
                except Exception:
                    pass
            elif source is not None:
                try:
                    source.release()
                except Exception:
                    pass
            self._set_status('KHONG MO DUOC CAMERA')
            self._log('Loi mo camera: %s' % exc)
            return

        try:
            self._grabber = grabber
            self._camera_stopping.clear()
            self._stop_event.clear()
            self._pipeline_meter.reset()
            self._fps = 0.0
            self._ensure_preview_thread()
            self._thread = threading.Thread(target=self._loop, args=(grabber,))
            self._thread.daemon = True
            self._thread.start()
        except Exception as exc:
            self._camera_stopping.set()
            self._grabber = None
            try:
                grabber.stop()
            except Exception:
                pass
            self._set_status('KHONG KHOI DONG DUOC THREAD CAMERA')
            self._log('Loi khoi dong vong camera: %s' % exc)
            return
        self._set_status('camera dang chay (chua ARM - xe khong chay)')
        self._log('Camera OK, backend=%s'
                  % getattr(source, 'backend', self.source_kind))

    def _on_run(self, _b=None):
        with self._lifecycle_lock:
            return self._on_run_locked(_b)

    def _on_run_locked(self, _b=None):
        if (self._grabber is None or self._camera_stopping.is_set() or
                getattr(self._grabber, 'eof', False) or
                getattr(self._grabber, 'error', None) is not None):
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
            self.mode = MODE_AUTO
            with self._engine_lock:
                self.engine.start_run()
            with self._state_lock:
                self._cv_steer = 0.0
                self._cv_throttle = 0.0
                self._cv_started_mono = None
            if not self._start_run_log(MODE_AUTO):
                self.mode = MODE_STOP
                with self._engine_lock:
                    self.engine.stop_run()
                self._set_status('TU CHOI CHAY - khong tao duoc log')
                return
            self._armed = True
            self._ensure_control_thread()
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
            self.mode = MODE_STOP
            with self._engine_lock:
                self.engine.stop_run()
            self._finish_run_log('start_error')
            self._set_status('LOI DRIVER - xe khong chay')
            self._log('Khong chay duoc voi driver %s: %s' % (self.driver_kind, exc))

    def _disarm(self, reason='stop'):
        with self._lifecycle_lock:
            return self._disarm_locked(reason)

    def _disarm_locked(self, reason='stop'):
        self._armed = False
        self.mode = MODE_STOP
        # Cat phan cung TRUOC khi cho inference/TensorRT tra engine lock. Nut
        # DUNG KHAN CAP khong duoc giu ga cu chi vi mot frame dang bi spike/treo.
        if self._driver is not None:
            with self._driver_lock:
                try:
                    self._driver.stop()
                except Exception:
                    pass
        # Detach + flush log ngay sau khi phan cung da neutral. Khong de mot
        # inference treo giu engine_lock lam nut DUNG tra ve ma file van mo.
        self._finish_run_log(reason)
        # Chi gan _run_t0=None (atomic tren CPython). Khong cho DUNG cho
        # TensorRT/inference; frame dang xu ly se bi `_armed=False` loai bo,
        # con lan CHAY sau van doi engine_lock va reset controller day du.
        self.engine.stop_run()
        self.shaper.reset()
        self.btn_manual.description = 'LAI TAY (thu data)'

    def _on_disarm(self, _b=None):
        self._disarm('stop')
        self._set_status('DA DUNG - motor=0')
        self._log('Da dung xe.')

    def _on_emergency(self, _b=None):
        self._disarm('emergency_stop')
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
    def _nguon_banner(self):
        w = self.widgets
        return w.HTML(
            '<div style="padding:6px 10px;margin-bottom:6px;border-radius:4px;'
            'font-weight:bold;background:%s;color:#fff">NGUON BAM VACH: %s</div>'
            % (('#1b7f3b', 'MODEL CNN (TensorRT) - models/lane_tiny.engine')
               if self.dung_cnn else
               ('#8a6d1f', 'CV CO DIEN (nguong mau) - KHONG dung model')))

    # --- anh xa cho num CUA SOM ------------------------------------------
    # 0.00 = cua muon, nhe nhang   |   1.00 = xong vao cua som va manh
    # Mot num viet hai khoa vi chung luon phai di cung nhau: phat hien cua som
    # (curve_enter thap) ma khong be lai som (feedforward thap) thi chi duoc
    # bo ga som, xe van cat cua.
    @staticmethod
    def _cua_som_to_cfg(p):
        return {
            'control.curve_enter': round(0.30 - 0.24 * p, 3),
            'control.curve_exit': round((0.30 - 0.24 * p) * 0.65, 3),
            'control.curve_feedforward': round(0.60 + 1.60 * p, 3),
        }

    # --- anh xa cho num PHANH TRUOC CUA -----------------------------------
    # 0.00 = gan nhu khong phanh (giu toc do vao cua)
    # 1.00 = phanh manh khi thay cua
    @staticmethod
    def _phanh_to_cfg(p):
        return {
            'control.curve_brake': round(0.70 * p, 3),   # theo do cong (don dau)
            'control.slowdown': round(0.05 + 0.35 * p, 3),  # theo cte (phan ung)
        }

    @staticmethod
    def _cfg_to_phanh(curve_brake):
        return max(0.0, min(1.0, float(curve_brake) / 0.70))

    @staticmethod
    def _cfg_to_cua_som(curve_enter):
        return max(0.0, min(1.0, (0.30 - float(curve_enter)) / 0.24))

    def _build_simple_panel(self):
        """Bon num. Moi num viet mot NHOM khoa di lien voi nhau."""
        w = self.widgets
        g = self.engine.get_param
        # Nhan hep lai va slider ngan lai de o doc so KHONG bi cot phai cat mat.
        # Truoc day nhan 150px + slider 440px bi tran ra ngoai cot, nguoi dung
        # chi thay mot chu so o ria.
        L = dict(style={'description_width': '128px'},
                 layout=w.Layout(width='340px'),
                 readout_format='.2f', continuous_update=False)

        # Dai noi rong de dua toc do. Tran an toan that su van la
        # `control.v_max`, tu nang theo hai num nay.
        toc_do = w.FloatSlider(value=float(g('control.v_straight', 0.30)),
                               min=0.05, max=1.00, step=0.01,
                               description='TOC DO thang', **L)
        toc_cua = w.FloatSlider(value=float(g('control.v_corner', 0.12)),
                                min=0.05, max=1.00, step=0.01,
                                description='TOC DO trong cua', **L)
        cua_som = w.FloatSlider(
            value=self._cfg_to_cua_som(g('control.curve_enter', 0.12)),
            min=0.0, max=1.0, step=0.05, description='CUA SOM', **L)
        phanh = w.FloatSlider(
            value=self._cfg_to_phanh(g('control.curve_brake', 0.30)),
            min=0.0, max=1.0, step=0.05,
            description='PHANH TRUOC CUA', **L)
        goc_lai = w.FloatSlider(
            value=float(g('control.driver.steering_output_max', 0.40)),
            min=0.30, max=1.00, step=0.05,
            description='GOC LAI TOI DA', **L)
        # Moi num mot dong so RIENG ngay duoi. Doc dam, luon hien du cot hep.
        notes = dict((k, w.HTML()) for k in
                     ('toc_do', 'toc_cua', 'cua_som', 'phanh', 'goc_lai'))
        info = w.HTML()

        def _n(html):
            return ('<div style="margin:-8px 0 10px 132px;font-size:11px;'
                    'color:#333;line-height:1.4">%s</div>' % html)

        def _cap_v_max():
            # Tran ga phai >= ca hai muc ga, neu khong no cat mat muc cao hon
            # va num kia nhin nhu hong.
            hi = max(float(self.engine.get_param('control.v_straight', 0.0)),
                     float(self.engine.get_param('control.v_corner', 0.0)))
            self.engine.set_param('control.v_max', round(min(1.0, hi * 1.25), 3))

        def _refresh():
            gp = self.engine.get_param
            notes['toc_do'].value = _n(
                'ga doan thang = <b>%.2f</b> &nbsp; (tran ga tu nang: '
                '<b>%.2f</b>)' % (gp('control.v_straight', 0),
                                  gp('control.v_max', 0)))
            notes['toc_cua'].value = _n(
                'ga trong cua = <b>%.2f</b> &nbsp; (= <b>%.0f%%</b> ga thang)'
                % (gp('control.v_corner', 0),
                   100.0 * float(gp('control.v_corner', 0))
                   / max(float(gp('control.v_straight', 1)), 1e-6)))
            notes['cua_som'].value = _n(
                'vao cua khi do cong &ge; <b>%.2f</b> &nbsp;|&nbsp; be lai som '
                '<b>%.2f</b>' % (gp('control.curve_enter', 0),
                                 gp('control.curve_feedforward', 0)))
            notes['phanh'].value = _n(
                'phanh theo do cong <b>%.2f</b> &nbsp;|&nbsp; theo do lech '
                '<b>%.2f</b>' % (gp('control.curve_brake', 0),
                                 gp('control.slowdown', 0)))
            tran_sv = float(gp('control.driver.steering_output_max', 0))
            notes['goc_lai'].value = _n(
                'tran servo = <b>%.2f</b> &nbsp; (<b>%.0f%%</b> tam lai)'
                % (tran_sv, 100.0 * tran_sv))
            # Neu lai da cham tran, tang toc chi lam xe chay rong hon. Bao ngay
            # o day thay vi de nguoi dung tu doc chu HARD-LIMIT tren panel.
            tran = float(gp('control.driver.steering_output_max', 0.0))
            canh = ''
            if float(gp('control.v_straight', 0)) > 0.45 and tran < 0.75:
                canh = ('<br><span style="color:#b00020">Ga cao ma TRAN SERVO '
                        'moi %.2f - neu panel hien <b>HARD-LIMIT</b> thi lai da '
                        'bao hoa, tang toc se chay rong ra ngoai cua.</span>'
                        % tran)
            info.value = ('<div style="margin-left:132px;font-size:11px">%s'
                          '</div>' % canh) if canh else ''

        def mk(fn):
            def _h(change):
                with self._engine_lock:
                    fn(float(change['new']))
                    _cap_v_max()
                _refresh()
            return _h

        def set_toc_do(v):
            self.engine.set_param('control.v_straight', v)

        def set_toc_cua(v):
            self.engine.set_param('control.v_corner', v)

        def set_cua_som(v):
            for k, val in self._cua_som_to_cfg(v).items():
                self.engine.set_param(k, val)

        def set_phanh(v):
            for k, val in self._phanh_to_cfg(v).items():
                self.engine.set_param(k, val)

        def set_goc_lai(v):
            # Doi CA HAI chieu; doi mot ben thi xe chi cua duoc mot huong.
            self.engine.set_param('control.driver.steering_output_max', v)
            self.engine.set_param('control.driver.steering_output_min', -v)

        for sl, fn in ((toc_do, set_toc_do), (toc_cua, set_toc_cua),
                       (cua_som, set_cua_som), (phanh, set_phanh),
                       (goc_lai, set_goc_lai)):
            sl.observe(mk(fn), names='value')
        _refresh()

        huong_dan = w.HTML(
            '<hr style="margin:12px 0">'
            '<div style="font-size:11px;line-height:1.6;max-width:480px">'
            '<b>TOC DO thang</b> &mdash; ga o doan thang. Diem thoi gian an o day.<br>'
            '<b>TOC DO trong cua</b> &mdash; ga khi dang cua. <b>Giam</b> neu xe '
            'vang ra NGOAI cua &mdash; chay cham thi ban kinh cua nho lai.<br>'
            '<b>CUA SOM</b> &mdash; xong vao cua som va manh den dau. <b>Tang</b> '
            'neu xe cua muon, cat vao phia TRONG cua. <b>Giam</b> neu doan thang '
            'xe luon zigzag.<br>'
            '<b>PHANH TRUOC CUA</b> &mdash; bo ga ngay khi NHIN THAY cua, '
            'truoc luc xe lech. <b>Tang</b> neu xe vao cua con nhanh qua. '
            '<b>Giam</b> neu xe phanh oan o doan thang.<br>'
            '<b>GOC LAI TOI DA</b> &mdash; banh be duoc toi dau. <b>Tang</b> neu '
            'cua khong noi, chay thang ra ngoai lane.<br>'
            '<span style="color:#b00020"><b>CANH BAO:</b> GOC LAI TOI DA la gioi '
            'han CO KHI. KE BANH KHOI MAT DAT, tang tung nac 0.05, nghe tieng '
            'servo. Nghe ken thi lui lai mot nac.</span></div>')
        return w.VBox([toc_do, notes['toc_do'],
                       toc_cua, notes['toc_cua'],
                       cua_som, notes['cua_som'],
                       phanh, notes['phanh'],
                       goc_lai, notes['goc_lai'],
                       info, huong_dan])

    def _widget_simple(self):
        """Chi camera + bam line + toc do. Dung CHUNG vong dieu khien va CHUNG
        co che an toan voi giao dien day du - o day chi bay it widget hon."""
        w = self.widgets
        that = self.driver_kind != 'dryrun'
        canh_bao = w.HTML(
            '<div style="padding:8px 10px;border-radius:4px;background:%s;'
            'color:#fff;font-weight:bold">%s</div>'
            % (('#b00020', 'XE SE CHAY THAT khi bam CHAY. Lan dau: KE BANH '
                           'KHOI MAT DAT.') if that else
               ('#0a7d3b', 'DRYRUN - banh KHONG quay du bam CHAY.')))
        return w.VBox([
            self._nguon_banner(),
            canh_bao,
            w.HBox([self.btn_open, self.btn_run, self.btn_halt, self.btn_stop]),
            w.HBox([self.btn_log]),
            w.HBox([self.preview,
                    w.VBox([w.HTML('<b style="font-size:14px">DIEU CHINH</b>'),
                            self._build_simple_panel(), self.btn_close])]),
            self.status, self.metrics, self.output,
        ])

    def widget(self):
        if self.simple:
            return self._widget_simple()
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
        row_tools = w.HBox([self.btn_log, self.btn_reset, self.btn_save,
                            self.btn_close])

        pad_box = w.VBox([
            self.controller_view,
            w.HBox([self.steering_axis, self.throttle_axis,
                    self.invert_steering, self.invert_throttle]),
            w.HBox([self.use_deadman, self.deadman_button]),
            w.HTML('<hr><b>Bo goc va toc do khi LAI TAY</b><br>'
                   '<small>Chi anh huong che do LAI TAY. Che do tu dong dung '
                   'cac num trong tab "Dieu khien".<br>'
                   '<b>Bo goc van bi hard-limit cua driver chan</b> giong che '
                   'do tu dong - xem muc 6 trong notebook.</small>'),
            self.manual_box,
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

        lane_tab = w.VBox([self._nguon_banner(), self.lane_box])

        tabs = w.Tab(children=[lane_tab, self.control_box, pad_box])
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
        self._camera_stopping.set()
        self._stop_event.set()
        self._control_stop.set()
        self._preview_stop.set()
        self._preview_event.set()
        self._disarm('ui_close')
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
        if self._preview_thread is not None and self._preview_thread.is_alive():
            self._preview_thread.join(timeout=2.0)
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
                     controller_index=0, data_root='data/driving',
                     overrides=None, simple=False):
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
                        data_root=data_root, overrides=overrides,
                        simple=simple).show()
