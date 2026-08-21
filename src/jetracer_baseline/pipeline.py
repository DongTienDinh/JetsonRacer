# -*- coding: utf-8 -*-
"""Vong dieu khien chinh + luong detect tach rieng.

Vi sao tach luong (DB §3.6 va §4.9):
  - Diem FPS la NHI PHAN: >= 20 FPS duoc 10 diem, 19.9 FPS duoc 0.
  - Neu detector nang chay dong bo trong vong chinh, FPS tut xuong ~8-12 tren
    Jetson Nano -> mat trang 10 diem VA bam lane giat.
  - Tach ra: vong chinh 30 Hz (lane CV ~3 ms), detect 10 Hz o luong rieng,
    ket qua duoc cache. Worst case latency bien bao:
        100 ms (cho chu ky detect) + ~40 ms (inference) + 33 ms (tick) ~ 175 ms
    < 300 ms yeu cau.

Trung thuc khi do FPS: `fps` la nhip cua vong chinh - vong nay CO doc anh, CO
dung ket qua nhan dien va CO sinh lenh dieu khien, dung dinh nghia DB §3.6.
`det_fps` duoc log rieng de BTC thay ro ca hai con so.
"""

import os
import threading
import time

import cv2
import numpy as np

from . import fsm as fsm_mod
from .camera import (LatestFrameGrabber, SyncGrabber,
                     shading_applied_at_source)
from .control.corner import CornerController
from .control.driver import build_driver
from .control.pid import PID
from .logging_csv import RunLogger
from .perception import build_lane_detector
from .perception.lane import LaneDetector  # noqa: F401  giu cho code cu import
from .perception.shading import ShadingCorrector
from .perception.signs import SignTracker, build_detector
from .perception.stopline import StoplineDetector
from .recorder import FrameRecorder


class _DetectWorker(object):
    """Chay detector o luong rieng, luon cache ket qua moi nhat."""

    def __init__(self, detector, grabber, hz):
        self.detector = detector
        self.grabber = grabber
        self.period = 1.0 / max(1.0, float(hz))
        self._lock = threading.Lock()
        self._result = []
        self._result_seq = 0
        self._fps = 0.0
        self._mean_interval = None
        self._last_completed = None
        self._last_frame_id = None
        self._stopped = False
        self._thread = threading.Thread(target=self._loop)
        self._thread.daemon = True

    def start(self):
        self._thread.start()
        return self

    def _loop(self):
        intervals = []
        while not self._stopped:
            t0 = time.monotonic()
            # peek: khong duoc nhay frame cua vong chinh o che do replay
            frame, frame_id = self.grabber.peek()
            completed = None
            if frame is not None and frame_id != self._last_frame_id:
                detector_ok = True
                try:
                    dets = self.detector.detect(frame)
                except Exception:
                    dets = []
                    detector_ok = False
                with self._lock:
                    self._result = dets
                    # Moi lan inference hoan tat/that bai la mot quan sat moi.
                    # Runner chi duoc vote bien bao mot lan cho moi sequence.
                    self._result_seq += 1
                self._last_frame_id = frame_id
                if not detector_ok:
                    with self._lock:
                        self._fps = 0.0
                    dt = time.monotonic() - t0
                    sleep = self.period - dt
                    if sleep > 0:
                        time.sleep(sleep)
                    continue
                completed = time.monotonic()
                if self._last_completed is not None:
                    dt_complete = completed - self._last_completed
                    if dt_complete > 0.0:
                        intervals.append(dt_complete)
                        if len(intervals) > 30:
                            intervals.pop(0)
                self._last_completed = completed
                if intervals:
                    with self._lock:
                        self._mean_interval = sum(intervals) / len(intervals)
                        self._fps = 1.0 / self._mean_interval
            dt = time.monotonic() - t0
            sleep = self.period - dt
            if sleep > 0:
                time.sleep(sleep)

    def get(self):
        with self._lock:
            fps = self._fps
            result = list(self._result)
            result_seq = self._result_seq
            mean_interval = self._mean_interval
        # Camera/detector dung thi khong duoc giu mai mot FPS dep tu qua khu.
        # Detector cham hon period van la detector khoe: timeout phai co gian
        # theo cadence hoan tat da do, khong ep cung 2.5/detect_hz.
        stale_after = max(
            2.5 * self.period,
            2.5 * mean_interval if mean_interval is not None else 0.0)
        if (self._last_completed is None or
                time.monotonic() - self._last_completed > stale_after):
            fps = 0.0
        return result, fps, result_seq

    def stop(self):
        self._stopped = True
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


class FpsMeter(object):
    def __init__(self, window=60):
        self.window = window
        self._times = []

    def tick(self, dt):
        self._times.append(dt)
        if len(self._times) > self.window:
            self._times.pop(0)

    @property
    def fps(self):
        if not self._times:
            return 0.0
        mean_dt = sum(self._times) / len(self._times)
        return 1.0 / mean_dt if mean_dt > 0 else 0.0


class MotionEstimator(object):
    """Uoc luong xe co dang di chuyen khong, bang chenh lech anh lien tiep.

    JetRacer khong chac co encoder banh xe. Chenh lech anh la tin hieu san co:
    xe ket -> anh gan nhu dung yen du throttle > 0. Dung de tu thoat ket TRUOC
    moc 10 giay cua DB §6 (dong ho khong dung trong luc restart).
    """

    def __init__(self, threshold=2.0, size=(64, 48)):
        self.threshold = threshold
        self.size = size
        self._prev = None

    def update(self, frame_bgr):
        small = cv2.cvtColor(cv2.resize(frame_bgr, self.size), cv2.COLOR_BGR2GRAY)
        small = small.astype(np.float32)
        if self._prev is None:
            self._prev = small
            return True, 0.0
        diff = float(np.mean(np.abs(small - self._prev)))
        self._prev = small
        return diff >= self.threshold, diff


class Runner(object):
    def __init__(self, cfg, task='speed', driver_kind='dryrun', source=None,
                 detector_script=None, log_dir=None, verbose=True, sync=False,
                 realtime=True, record_dir=None, record_every=1):
        self.cfg = cfg
        self.task = task
        self.verbose = verbose
        # Gan truoc de close() co the don dep an toan neu constructor loi giua
        # chung (camera/driver/model/log khong duoc bi giu den lan chay sau).
        self.grabber = None
        self.worker = None
        self.driver = None
        self.logger = None
        self.recorder = None
        # Cung mot run_name cho ca CSV va video -> ghep duoc theo ten file.
        self.run_name = time.strftime('%Y%m%d_%H%M%S')
        # sync=True  : replay offline, xu ly du moi frame, tat dinh
        # sync=False : camera that, vut frame cu de khong lai theo qua khu
        self.sync = sync
        # realtime=False: replay chay het toc do, khong sleep giu nhip
        self.realtime = realtime

        grabber_cls = SyncGrabber if sync else LatestFrameGrabber
        try:
            self.grabber = grabber_cls(source).start()
        except Exception:
            try:
                source.release()
            except Exception:
                pass
            raise
        startup_error = getattr(self.grabber, 'error', None)
        if startup_error is not None:
            self.grabber.stop()
            raise RuntimeError('Camera/source khong khoi dong duoc: %s' %
                               startup_error)
        try:
            # Sua lens shading mau TRUOC moi khau nhan dien. Camera CSI cua xe
            # do gap ~1.6 lan o goc anh so voi tam; nguon da tu sua mau thi
            # khong duoc sua lan hai trong pipeline.
            if shading_applied_at_source(cfg):
                self.shading = ShadingCorrector.disabled()
            else:
                self.shading = ShadingCorrector.from_config(cfg)
            self.proc_size = (int(cfg.get('pipeline.proc_width', 320)),
                              int(cfg.get('pipeline.proc_height', 240)))
            self.lane = build_lane_detector(cfg)
            self.stopline = StoplineDetector(cfg)
            self.tracker = (SignTracker(cfg)
                            if self.task == 'smartcity' else None)
            # Speed Track khong dung bien bao. Smart City tao worker o day
            # nhung chi start sau khi TAT CA resource khac da khoi tao thanh cong.
            self.detector = None
            if self.task == 'smartcity':
                self.detector = build_detector(cfg, script=detector_script)
                self.worker = _DetectWorker(
                    self.detector, self.grabber,
                    cfg.get('pipeline.detect_hz', 10))
        except Exception:
            self.close()
            raise

        try:
            self.pid = PID(
                cfg.get('control.pid.kp', 0.6),
                cfg.get('control.pid.ki', 0.0),
                cfg.get('control.pid.kd', 0.1),
                out_limit=cfg.get('control.pid.out_limit', 1.0),
            )
            # Cung mot bo dieu khien voi giao dien tune -> tune xong chay that
            # khong bi lech hanh vi.
            self.corner = CornerController(cfg)
            self.driver = build_driver(driver_kind, cfg)
            self.fsm = fsm_mod.DecisionFSM(cfg, task=task)
            self.motion = MotionEstimator(
                threshold=float(cfg.get('fsm.motion_threshold', 2.0)),
            )
            self.fps_meter = FpsMeter(cfg.get('pipeline.fps_window', 60))

            self.logger = RunLogger(
                log_dir or cfg.get('logging.dir', 'logs'),
                task,
                run_name=self.run_name,
                flush_every=cfg.get('logging.flush_every', 30),
            )

            self.record_every = max(1, int(record_every))
            if record_dir:
                if not os.path.isdir(record_dir):
                    os.makedirs(record_dir)
                # Lay stem THUC cua CSV (co the co _01 neu hai luot cung giay)
                # de video va log khong ghi de/lech ten nhau.
                log_stem = os.path.splitext(
                    os.path.basename(self.logger.path))[0]
                video_path = os.path.join(record_dir, log_stem + '.avi')
                self.recorder = FrameRecorder(
                    video_path,
                    fps=float(cfg.get('pipeline.control_hz', 30)))

            self.steer_look_w = float(
                cfg.get('control.steer_lookahead_weight', 0.5))
            self.steer_max = float(cfg.get('control.steer_max', 1.0))
            self.control_period = (
                1.0 / float(cfg.get('pipeline.control_hz', 30)))

            self._sign_reported = set()
            self._last_detection_seq = None
            self._stable_sign = None
            if self.worker is not None:
                self.worker.start()
        except Exception:
            self.close()
            raise

    def _update_sign_tracker(self, detections, result_seq, now):
        """Moi detector completion chi tao dung mot vote cho SignTracker."""
        if self.tracker is None:
            return None
        if result_seq != self._last_detection_seq:
            self._last_detection_seq = result_seq
            self._stable_sign = self.tracker.update(detections, now=now)
        return self._stable_sign

    def run(self, max_seconds=300.0, max_frames=None, auto_start=True):
        """max_seconds mac dinh 300 = dung gioi han 5 phut/luot cua BTC."""
        if auto_start:
            self.fsm.start()
        self.logger.mark_start()

        t_run0_mono = time.monotonic()
        t_prev_mono = t_run0_mono
        last_frame_id = None
        frames = 0
        pending_event = 'start'

        try:
            while True:
                loop_t0_mono = time.monotonic()
                loop_t0 = time.time()
                if (loop_t0_mono - t_run0_mono) >= max_seconds:
                    pending_event = 'timeout'
                    break
                if max_frames is not None and frames >= max_frames:
                    break

                frame, frame_id = self.grabber.read()
                if frame is None:
                    if self.grabber.eof:
                        pending_event = ('camera_error'
                                         if getattr(self.grabber, 'error', None)
                                         is not None else 'eof')
                        break
                    time.sleep(0.005)
                    continue
                if frame_id == last_frame_id:
                    # LatestFrameGrabber tra frame cache neu camera chua co anh
                    # moi. Xu ly lai no se tao FPS 30 gia trong khi camera co the
                    # dang chi ra 10 FPS (hoac da treo o mot khung hinh).
                    if self.grabber.eof:
                        pending_event = ('camera_error'
                                         if getattr(self.grabber, 'error', None)
                                         is not None else 'eof')
                        break
                    time.sleep(0.001)
                    continue
                last_frame_id = frame_id

                if self.recorder is not None and frames % self.record_every == 0:
                    # Frame THO, truoc resize - LaneDetector tu resize ben
                    # trong, giu nguyen o day de con tune duoc proc_width/height.
                    self.recorder.submit(frame, frame_id)

                # ---- Perception -------------------------------------------
                # Frame THO da duoc ghi o tren; tu day tro di dung frame da sua
                # mau. Ghi tho co chu dich: con hieu chuan lai shading duoc tu
                # video cu, con neu ghi anh da sua thi mat goc, khong lam lai duoc.
                proc = self.shading.apply_resized(frame, self.proc_size)
                lane_res = self.lane.process(proc)
                stop_res = self.stopline.process(proc) if self.task == 'smartcity' else None
                moving, _ = self.motion.update(proc)

                if self.worker is not None:
                    dets, det_fps, det_seq = self.worker.get()
                    stable = self._update_sign_tracker(
                        dets, det_seq, now=loop_t0)
                else:
                    det_fps, stable = 0.0, None

                # ---- Decision ---------------------------------------------
                cmd = self.fsm.step(lane_res, stopline=stop_res, sign=stable,
                                    moving=moving, now=loop_t0)

                # ---- Control ----------------------------------------------
                dt = loop_t0_mono - t_prev_mono
                t_prev_mono = loop_t0_mono
                # FPS = nhip THUC TE cua vong chinh (khoang cach giua hai lan lap),
                # KHONG phai 1/thoi-gian-xu-ly. Neu do bang thoi gian xu ly thi con
                # so bi thoi phong nhieu lan va khong the bao ve truoc BTC.
                if frames > 0:
                    self.fps_meter.tick(max(1e-6, dt))
                # Tron lech tai muc xe voi lech tai diem ngam xa. Chi dung
                # cte thi xe luon sua MUON o cua: den luc CTE tang du de PID
                # phan ung thi xe da an vao mep lane roi.
                err = ((1.0 - self.steer_look_w) * lane_res.cte
                       + self.steer_look_w * lane_res.cte_lookahead)
                pid_out = self.pid.step(err, dt)
                steer, throttle, drive_mode = self.corner.step(
                    pid_out, lane_res.cte, lane_res.curvature, dt,
                    lane_found=lane_res.found)

                if cmd.steer_override is not None:
                    steer = max(-self.steer_max,
                                min(self.steer_max, cmd.steer_override))
                    self.pid.reset()
                if cmd.force_stop:
                    throttle = 0.0
                else:
                    throttle *= cmd.throttle_scale
                self.driver.set(steer, throttle)

                # ---- Log ---------------------------------------------------
                # latency = thoi gian xu ly 1 frame (doc anh -> sinh lenh), khac FPS
                latency_ms = (time.monotonic() - loop_t0_mono) * 1000.0

                sign_latency = None
                detected, conf = '', None
                if stable is not None:
                    label, conf, first_seen = stable
                    detected = label
                    if label not in self._sign_reported:
                        # DB §4.9: tu luc bien hien ro den luc sinh lenh tuong ung
                        sign_latency = (time.time() - first_seen) * 1000.0
                        self._sign_reported.add(label)
                elif lane_res.found:
                    detected = 'lane'

                self.logger.write(
                    timestamp=loop_t0,
                    frame_id=frame_id,
                    fps=self.fps_meter.fps,
                    det_fps=det_fps,
                    latency_ms=latency_ms,
                    sign_latency_ms=sign_latency,
                    detected_object=detected,
                    confidence=conf,
                    decision=cmd.decision,
                    control_output='steer=%.3f;throttle=%.3f' % (steer, throttle),
                    cte=lane_res.cte,
                    cte_lookahead=lane_res.cte_lookahead,
                    curvature=lane_res.curvature,
                    lane_found=lane_res.found,
                    n_bands=lane_res.n_bands,
                    throttle=throttle,
                    drive_mode=drive_mode,
                    state=cmd.state,
                    event=(pending_event or cmd.event),
                )
                pending_event = ''
                frames += 1

                # Giu nhip vong chinh on dinh (bo qua khi replay khong realtime)
                if self.realtime:
                    sleep = self.control_period - (time.monotonic() - loop_t0_mono)
                    if sleep > 0:
                        time.sleep(sleep)
        except KeyboardInterrupt:
            pending_event = 'keyboard_interrupt'
        except Exception:
            pending_event = 'pipeline_error'
            raise
        finally:
            run_stopped_mono = time.monotonic()
            run_stopped_wall = time.time()
            # An toan va bang chung doc lap nhau: driver loi luc stop khong duoc
            # ngan dong terminal/flush log; log loi cung khong duoc ngan cat ga.
            try:
                self.driver.stop()
            except Exception as exc:
                if self.verbose:
                    print('CANH BAO: loi dung driver: %s' % exc)
            try:
                self.logger.write(
                    timestamp=run_stopped_wall, frame_id=frames,
                    fps=self.fps_meter.fps, decision='stop', state='FINISHED',
                    event=pending_event or 'stop')
            except Exception as exc:
                if self.verbose:
                    print('CANH BAO: khong ghi duoc dong ket log: %s' % exc)
            run_seconds = run_stopped_mono - t_run0_mono
            summary = {
                'frames': frames,
                'seconds': run_seconds,
                # fps_mean = frames/seconds CA LUOT - dung con so nay de doi
                # chieu nguong 20 cua BTC. fps_window = gia tri truot 60-frame
                # cuoi (giong het cot 'fps' trong CSV) - chi de xem nhip luc
                # ket thuc, KHONG dung de danh gia dat/khong dat.
                'fps_mean': (frames / run_seconds) if run_seconds > 0 else 0.0,
                'fps_window': self.fps_meter.fps,
                'log': self.logger.path,
            }
            self.close()
            # Doc sau close() de queue cua recorder da drain xong.
            if self.recorder is not None:
                summary['record_written'] = self.recorder.n_written
                summary['record_dropped'] = self.recorder.n_dropped
                summary['record_path'] = self.recorder.path
            else:
                summary['record_written'] = None
                summary['record_dropped'] = None
                summary['record_path'] = None

        if self.verbose:
            print('Frames        : %d' % summary['frames'])
            print('Thoi gian     : %.2f s' % summary['seconds'])
            if self.realtime:
                print('FPS trung binh (ca luot): %.2f  (%s nguong 20 cua BTC)' % (
                    summary['fps_mean'],
                    'DAT' if summary['fps_mean'] >= 20 else 'CHUA DAT'))
                print('FPS truot (60 frame cuoi): %.2f' % summary['fps_window'])
            else:
                print('FPS trung binh: %.2f  [replay khong realtime - KHONG dung'
                      ' con so nay de doi chieu nguong 20 FPS]' % summary['fps_mean'])
            print('Log           : %s' % summary['log'])
            if summary['record_path'] is not None:
                print('Video         : %s  (ghi %d, vut %d)' % (
                    summary['record_path'], summary['record_written'],
                    summary['record_dropped']))
        return summary

    def close(self):
        resources = []
        if self.worker is not None:
            resources.append(('detect worker', self.worker.stop))
        if self.grabber is not None:
            resources.append(('camera grabber', self.grabber.stop))
        if self.driver is not None:
            resources.append(('driver', self.driver.close))
        if self.logger is not None:
            resources.append(('logger', self.logger.close))
        if self.recorder is not None:
            resources.append(('recorder', self.recorder.close))
        for name, close_fn in resources:
            try:
                close_fn()
            except Exception as exc:
                if self.verbose:
                    print('CANH BAO: loi dong %s: %s' % (name, exc))
