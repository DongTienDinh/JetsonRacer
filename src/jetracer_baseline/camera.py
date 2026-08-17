# -*- coding: utf-8 -*-
"""Nguon anh: CSI camera tren Jetson, file video (replay), hoac anh tong hop.

Diem quan trong: luong capture LUON giu frame moi nhat va vut frame cu.
Neu dung queue, vong dieu khien se xu ly anh tre -> xe lai theo qua khu.
"""

import glob
import os
import tempfile
import threading
import time

import cv2
import numpy as np


try:
    import fcntl
except ImportError:  # Windows: chi dung cho smoke-test, khong co camera CSI.
    fcntl = None


CAMERA_LOCK_PATH = os.path.join(tempfile.gettempdir(), 'jetracer_camera.lock')


class CameraLease(object):
    """Khoa lien tien trinh de hai notebook cua project khong mo CSI cung luc."""

    def __init__(self, path=CAMERA_LOCK_PATH):
        self.path = path
        self._file = None

    def acquire(self):
        if fcntl is None:
            return
        handle = open(self.path, 'a+')
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, OSError):
            handle.seek(0)
            owner = handle.read().strip() or 'khong ro PID'
            handle.close()
            raise IOError(
                'Camera dang bi mot tien trinh JetsonRacer khac giu (%s). '
                'Dong kernel/notebook cu truoc khi mo lai.' % owner)
        handle.seek(0)
        handle.truncate()
        handle.write('pid=%d cwd=%s' % (os.getpid(), os.getcwd()))
        handle.flush()
        self._file = handle

    def release(self):
        if self._file is None:
            return
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            self._file.close()
        except Exception:
            pass
        self._file = None


def gstreamer_pipeline(width, height, fps, flip_method=0,
                       capture_width=None, capture_height=None, sensor_id=0):
    """Pipeline nvarguscamerasrc cho camera CSI (IMX219) tren Jetson Nano."""
    capture_width = int(capture_width or width)
    capture_height = int(capture_height or height)
    return (
        'nvarguscamerasrc sensor-id={sensor_id} ! '
        'video/x-raw(memory:NVMM), width=(int){cw}, height=(int){ch}, '
        'format=(string)NV12, framerate=(fraction){fps}/1 ! '
        'nvvidconv flip-method={flip} ! '
        'video/x-raw, width=(int){w}, height=(int){h}, format=(string)BGRx ! '
        'videoconvert ! video/x-raw, format=(string)BGR ! '
        'appsink drop=true max-buffers=1 sync=false'
    ).format(sensor_id=int(sensor_id), cw=capture_width, ch=capture_height,
             fps=fps, flip=flip_method, w=width, h=height)


def camera_environment_report():
    """Thong tin chi doc de hien thi khi camera Jetson khong mo duoc."""
    info = cv2.getBuildInformation()
    gstreamer = 'UNKNOWN'
    for line in info.split('\n'):
        if line.strip().startswith('GStreamer'):
            gstreamer = line.split(':', 1)[-1].strip()
            break
    return {
        'opencv': getattr(cv2, '__version__', '?'),
        'gstreamer': gstreamer,
        'video_devices': sorted(glob.glob('/dev/video*')),
    }


def format_camera_environment(report=None):
    report = report or camera_environment_report()
    devices = ','.join(report.get('video_devices') or []) or '(khong co)'
    return 'OpenCV=%s, GStreamer=%s, devices=%s' % (
        report.get('opencv', '?'), report.get('gstreamer', '?'), devices)


class FrameSource(object):
    """Interface chung."""

    def read(self):
        """Tra ve (ok, frame_BGR)."""
        raise NotImplementedError

    def release(self):
        pass


class SyntheticSource(FrameSource):
    """Sinh anh gia lap duong dua: nen toi, vach giua trang dut khuc, co uon luon.

    Dung de smoke-test toan bo pipeline tren laptop khi chua co xe va chua co video.
    KHONG phai simulator vat ly - chi de kiem tra pipeline khong vo, khong thay the
    thu nghiem tren sa ban that.
    """

    def __init__(self, width=640, height=480, n_frames=None):
        self.width = width
        self.height = height
        self.n_frames = n_frames
        self._i = 0

    def read(self):
        if self.n_frames is not None and self._i >= self.n_frames:
            return False, None

        h, w = self.height, self.width
        img = np.full((h, w, 3), 35, dtype=np.uint8)

        # Duong dua uon theo sin, lech dan theo thoi gian
        phase = self._i * 0.05
        for y in range(int(h * 0.5), h):
            depth = (y - h * 0.5) / (h * 0.5)
            center = int(w * 0.5 + np.sin(phase + depth * 1.2) * w * 0.18)
            half = int(w * 0.16 + depth * w * 0.14)
            x0 = max(0, center - half)
            x1 = min(w, center + half)
            img[y, x0:x1] = (60, 60, 60)          # mat lane sang hon nen
            # vach giua trang dut khuc
            if (y + self._i * 6) % 40 < 20:
                img[y, max(0, center - 2):min(w, center + 2)] = (235, 235, 235)

        self._i += 1
        return True, img

    def release(self):
        pass


class VideoSource(FrameSource):
    """Doc tu file video - dung cho replay offline (dev khong can xe)."""

    def __init__(self, path, loop=False):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError('Khong mo duoc video: ' + str(path))
        self.loop = loop

    def read(self):
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return ok, frame

    def release(self):
        self.cap.release()


class CSISource(FrameSource):
    """Camera CSI tren Jetson qua GStreamer."""

    def __init__(self, width, height, fps, flip_method=0,
                 allow_usb_fallback=True, usb_index=0,
                 startup_attempts=15, startup_delay_s=0.10,
                 capture_width=1280, capture_height=720, sensor_id=0,
                 csi_open_attempts=2, csi_retry_delay_s=0.50):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.flip_method = int(flip_method)
        self.allow_usb_fallback = bool(allow_usb_fallback)
        self.usb_index = int(usb_index)
        self.startup_attempts = max(1, int(startup_attempts))
        self.startup_delay_s = max(0.0, float(startup_delay_s))
        self.capture_width = int(capture_width)
        self.capture_height = int(capture_height)
        self.sensor_id = int(sensor_id)
        self.csi_open_attempts = max(1, int(csi_open_attempts))
        self.csi_retry_delay_s = max(0.0, float(csi_retry_delay_s))
        self.pipeline = gstreamer_pipeline(
            self.width, self.height, self.fps, self.flip_method,
            capture_width=self.capture_width,
            capture_height=self.capture_height,
            sensor_id=self.sensor_id)
        self.backend = 'csi-gstreamer'
        self.startup_notes = []
        self.last_error = None
        self._needs_warmup = True
        self.cap = None
        self._lease = CameraLease()

        self._lease.acquire()
        try:
            opened = self._open_csi_with_retry()
            if not opened:
                reason = 'OpenCV khong open duoc pipeline nvarguscamerasrc'
                if self.allow_usb_fallback:
                    self._switch_to_usb(reason)
                else:
                    raise IOError(self._failure_message(reason))
        except Exception:
            self.release()
            raise

    def _open_csi_capture(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.backend = 'csi-gstreamer'
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        return self.cap.isOpened()

    def _open_csi_with_retry(self):
        for attempt in range(1, self.csi_open_attempts + 1):
            if self._open_csi_capture():
                return True
            self.startup_notes.append(
                'CSI open lan %d/%d that bai' %
                (attempt, self.csi_open_attempts))
            if attempt < self.csi_open_attempts and self.csi_retry_delay_s > 0:
                time.sleep(self.csi_retry_delay_s)
        return False

    def _failure_message(self, reason):
        return (
            '%s. CSI sensor=%d capture=%dx%d@%d -> output=%dx%d. %s. '
            'Loi `Failed to create CaptureSession` thuong do kernel/tien trinh '
            'khac dang giu camera, nvargus-daemon bi ket, hoac cap CSI. '
            'va thu `sudo systemctl restart nvargus-daemon` sau khi da dong '
            'tat ca notebook camera.' % (
                reason, self.sensor_id, self.capture_width,
                self.capture_height, self.fps, self.width, self.height,
                format_camera_environment()))

    def _switch_to_usb(self, reason):
        try:
            self.cap.release()
        except Exception:
            pass
        self.startup_notes.append(
            'CSI loi (%s) -> thu USB/V4L2 index %d' % (reason, self.usb_index))
        self.cap = cv2.VideoCapture(self.usb_index)
        self.backend = 'usb-v4l2-index-%d' % self.usb_index
        if not self.cap.isOpened():
            self.cap.release()
            raise IOError(self._failure_message(
                'CSI va USB/V4L2 index %d deu khong open duoc' % self.usb_index))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._needs_warmup = True

    def _read_with_warmup(self):
        last_exception = None
        for _ in range(self.startup_attempts):
            try:
                ok, frame = self.cap.read()
            except Exception as exc:
                ok, frame = False, None
                last_exception = exc
            if ok and frame is not None:
                self.last_error = None
                return True, frame
            if self.startup_delay_s > 0:
                time.sleep(self.startup_delay_s)
        detail = 'khong co frame sau %d lan doc' % self.startup_attempts
        if last_exception is not None:
            detail += ' (%s)' % last_exception
        self.last_error = detail
        return False, None

    def read(self):
        if self._needs_warmup:
            self._needs_warmup = False
            ok, frame = self._read_with_warmup()
            if ok:
                return ok, frame
            # Argus doi khi bao open=True nhung CaptureSession chua tao duoc.
            # Release va mo lai CSI mot lan truoc khi chuyen sang USB.
            if self.backend == 'csi-gstreamer' and self.csi_open_attempts > 1:
                initial_error = self.last_error
                for attempt in range(2, self.csi_open_attempts + 1):
                    self.startup_notes.append(
                        'CSI co open nhung khong co frame; mo lai lan %d/%d' %
                        (attempt, self.csi_open_attempts))
                    if self.csi_retry_delay_s > 0:
                        time.sleep(self.csi_retry_delay_s)
                    if self._open_csi_capture():
                        ok, frame = self._read_with_warmup()
                        if ok:
                            return ok, frame
                self.last_error = initial_error
            if self.backend == 'csi-gstreamer' and self.allow_usb_fallback:
                self._switch_to_usb(self.last_error)
                self._needs_warmup = False
                ok, frame = self._read_with_warmup()
                if ok:
                    return ok, frame
            raise IOError(self._failure_message(
                '%s (%s)' % (self.backend, self.last_error)))

        try:
            ok, frame = self.cap.read()
        except Exception as exc:
            self.last_error = '%s read exception: %s' % (self.backend, exc)
            raise IOError(self.last_error)
        if not ok or frame is None:
            self.last_error = '%s da mo nhung mat frame trong luc chay' % self.backend
            return False, None
        return True, frame

    def release(self):
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        finally:
            self._lease.release()


class SyncGrabber(object):
    """Doc tuan tu, KHONG bo frame - dung cho replay offline.

    Vi sao can rieng: che do live phai vut frame cu (xe lai theo qua khu la nguy
    hiem), nhung khi replay video de lay so lieu cho paper thi phai xu ly DU moi
    frame, neu khong ket qua khong tat dinh va khong lap lai duoc.

    `read()`  : lay frame ke tiep (vong chinh)
    `peek()`  : xem frame hien tai, KHONG nhay toi (luong detect)
    """

    def __init__(self, source):
        self.source = source
        self._frame = None
        self._frame_id = 0
        self._eof = False
        self._lock = threading.Lock()

    def start(self):
        return self

    def read(self):
        ok, frame = self.source.read()
        if not ok:
            self._eof = True
            return None, self._frame_id
        with self._lock:
            self._frame = frame
            self._frame_id += 1
            return frame.copy(), self._frame_id

    def peek(self):
        with self._lock:
            if self._frame is None:
                return None, self._frame_id
            return self._frame.copy(), self._frame_id

    @property
    def eof(self):
        return self._eof

    def stop(self):
        self.source.release()


class LatestFrameGrabber(object):
    """Bao quanh mot FrameSource bang luong rieng, luon giu frame MOI NHAT.

    `read()` khong bao gio block cho I/O camera -> vong dieu khien giu duoc nhip
    on dinh, la dieu kien de dat FPS >= 20 (Diem FPS cua Speed Track).
    """

    def __init__(self, source, wait_timeout_s=5.0):
        self.source = source
        self.wait_timeout_s = max(0.05, float(wait_timeout_s))
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0
        self._stopped = False
        self._eof = False
        self._error = None
        self._thread = threading.Thread(target=self._loop)
        self._thread.daemon = True

    def start(self):
        self._thread.start()
        # Cho frame dau tien (toi da 5 s) de pipeline khong chay khong
        deadline = time.monotonic() + self.wait_timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._frame is not None or self._eof or self._error is not None:
                    return self
            time.sleep(0.005)
        with self._lock:
            if self._frame is None and self._error is None:
                self._error = IOError(
                    'Timeout %.1f s khi cho frame camera dau tien' %
                    self.wait_timeout_s)
        return self

    def _loop(self):
        while not self._stopped:
            try:
                ok, frame = self.source.read()
            except Exception as exc:
                with self._lock:
                    self._error = exc
                    self._eof = True
                break
            if not ok:
                detail = getattr(self.source, 'last_error', None)
                with self._lock:
                    if detail:
                        self._error = IOError(str(detail))
                    self._eof = True
                break
            with self._lock:
                self._frame = frame
                self._frame_id += 1

    def read(self):
        """Tra ve (frame, frame_id). frame = None neu het nguon."""
        with self._lock:
            if self._frame is None:
                return None, self._frame_id
            return self._frame.copy(), self._frame_id

    # O che do live, peek == read (deu tra ve frame moi nhat da cache).
    peek = read

    @property
    def eof(self):
        return self._eof

    @property
    def error(self):
        with self._lock:
            return self._error

    def stop(self):
        self._stopped = True
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)
        # VideoCapture.read() cua nvargus co the dang block. Release capture de
        # pha block, sau do cho thread them mot nhip de thoat sach.
        self.source.release()
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)


def build_source(cfg, kind, video_path=None, n_frames=None):
    """kind: 'csi' | 'video' | 'synthetic'."""
    if kind == 'video':
        if not video_path:
            raise ValueError('kind=video can --video <duong dan>')
        return VideoSource(video_path)
    if kind == 'synthetic':
        return SyntheticSource(
            width=cfg.get('camera.width', 640),
            height=cfg.get('camera.height', 480),
            n_frames=n_frames,
        )
    return CSISource(
        cfg.get('camera.width', 640),
        cfg.get('camera.height', 480),
        cfg.get('camera.fps', 30),
        flip_method=cfg.get('camera.flip_method', 0),
        allow_usb_fallback=cfg.get('camera.allow_usb_fallback', True),
        usb_index=cfg.get('camera.usb_index', 0),
        startup_attempts=cfg.get('camera.startup_attempts', 15),
        startup_delay_s=cfg.get('camera.startup_delay_s', 0.10),
        capture_width=cfg.get('camera.capture_width', 1280),
        capture_height=cfg.get('camera.capture_height', 720),
        sensor_id=cfg.get('camera.sensor_id', 0),
        csi_open_attempts=cfg.get('camera.csi_open_attempts', 2),
        csi_retry_delay_s=cfg.get('camera.csi_retry_delay_s', 0.50),
    )
