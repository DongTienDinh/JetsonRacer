# -*- coding: utf-8 -*-
"""Nguon anh: CSI camera tren Jetson, file video (replay), hoac anh tong hop.

Diem quan trong: luong capture LUON giu frame moi nhat va vut frame cu.
Neu dung queue, vong dieu khien se xu ly anh tre -> xe lai theo qua khu.
"""

import threading
import time

import cv2
import numpy as np


def gstreamer_pipeline(width, height, fps, flip_method=0):
    """Pipeline nvarguscamerasrc cho camera CSI (IMX219) tren Jetson Nano."""
    return (
        'nvarguscamerasrc ! '
        'video/x-raw(memory:NVMM), width=(int){cw}, height=(int){ch}, '
        'format=(string)NV12, framerate=(fraction){fps}/1 ! '
        'nvvidconv flip-method={flip} ! '
        'video/x-raw, width=(int){w}, height=(int){h}, format=(string)BGRx ! '
        'videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1'
    ).format(cw=width, ch=height, fps=fps, flip=flip_method, w=width, h=height)


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

    def __init__(self, width, height, fps, flip_method=0):
        pipeline = gstreamer_pipeline(width, height, fps, flip_method)
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            # Fallback: USB camera / dev khong co Jetson
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise IOError('Khong mo duoc camera (CSI lan fallback index 0)')
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


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

    def __init__(self, source):
        self.source = source
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0
        self._stopped = False
        self._eof = False
        self._thread = threading.Thread(target=self._loop)
        self._thread.daemon = True

    def start(self):
        self._thread.start()
        # Cho frame dau tien (toi da 5 s) de pipeline khong chay khong
        deadline = time.time() + 5.0
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None or self._eof:
                    return self
            time.sleep(0.005)
        return self

    def _loop(self):
        while not self._stopped:
            ok, frame = self.source.read()
            if not ok:
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

    def stop(self):
        self._stopped = True
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.source.release()


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
    )
