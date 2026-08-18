# -*- coding: utf-8 -*-
"""Ghi video trong luc `run`, khong duoc lam tut FPS vong chinh.

Boi canh (xem camera.py `LatestFrameGrabber.read`): `read()` KHONG tieu thu
frame - no tra ve frame moi nhat dang cache. Vong dieu khien nhanh hon camera
thi cung `frame_id` doc duoc nhieu lan; camera nhanh hon vong dieu khien thi
`frame_id` nhay coc. Vi vay:

  1. Chi ghi khi `frame_id` doi - ghi lap thi video co frame trung, timeline
     video giai sai so voi thoi gian that.
  2. Bat buoc co file sidecar (`video_index, frame_id, timestamp`) de ghep
     video voi CSV log khi phan tich sau nay. Khong co no thi frame thu N
     trong video KHONG tuong ung dong thu N trong CSV.

`submit()` khong bao gio block vong chinh: day vao queue gioi han, day roi thi
vut frame va tang `n_dropped`. Tha mat vai frame trong video con hon tut FPS
duoi 20 - diem FPS la nhi phan (DB §3.6), video chi phuc vu phan tich dinh tinh.
"""

import io
import os
import threading
import time
import csv

try:
    import Queue as queue  # Python 2 (khong dung nhung phong khi)
except ImportError:
    import queue

import cv2


class FrameRecorder(object):
    def __init__(self, path, fourcc='MJPG', fps=30.0, queue_size=30,
                 extra_fields=None):
        self.path = path
        self._fourcc = fourcc
        self._fps = float(fps)
        self._queue = queue.Queue(maxsize=queue_size)
        self._writer = None
        self._sidecar = None
        self._sidecar_csv = None
        self._extra_fields = list(extra_fields or [])
        self._video_index = 0
        self._last_frame_id = None
        self.n_written = 0
        self.n_dropped = 0
        self.error = None
        self._stopped = False
        self._thread = threading.Thread(target=self._loop)
        self._thread.daemon = True
        self._thread.start()

    def submit(self, frame, frame_id, timestamp=None, extra=None):
        """Khong bao gio block. Bo qua neu frame_id trung lan truoc."""
        if self._stopped or self.error is not None:
            return False
        if frame_id == self._last_frame_id:
            return False
        self._last_frame_id = frame_id
        try:
            self._queue.put_nowait((
                frame, frame_id,
                time.time() if timestamp is None else float(timestamp),
                dict(extra or {})))
            return True
        except queue.Full:
            self.n_dropped += 1
            return False

    def _open_writer(self, frame):
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*self._fourcc)
        self._writer = cv2.VideoWriter(self.path, fourcc, self._fps, (w, h))
        if not self._writer.isOpened():
            self._writer.release()
            self._writer = None
            raise IOError('Khong mo duoc VideoWriter: %s' % self.path)
        sidecar_path = os.path.splitext(self.path)[0] + '.sidecar.csv'
        self._sidecar = io.open(
            sidecar_path, 'w', newline='', encoding='utf-8')
        fields = ['video_index', 'frame_id', 'timestamp'] + self._extra_fields
        self._sidecar_csv = csv.DictWriter(self._sidecar, fieldnames=fields)
        self._sidecar_csv.writeheader()

    def _loop(self):
        while True:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self._stopped:
                    return
                continue
            frame, frame_id, ts, extra = item
            try:
                if self._writer is None:
                    self._open_writer(frame)
                self._writer.write(frame)
                row = {
                    'video_index': self._video_index,
                    'frame_id': frame_id,
                    'timestamp': '%.6f' % ts,
                }
                for name in self._extra_fields:
                    row[name] = extra.get(name, '')
                self._sidecar_csv.writerow(row)
                self._video_index += 1
                self.n_written += 1
            except Exception as exc:
                self.error = exc
                self._stopped = True
            finally:
                self._queue.task_done()
            if self.error is not None:
                # Khong de close() cho het timeout vi queue con ton sau loi codec.
                while True:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        break
                    else:
                        self._queue.task_done()
                return

    def close(self, drain_timeout=5.0):
        self._stopped = True
        deadline = time.time() + drain_timeout
        while not self._queue.empty() and time.time() < deadline:
            time.sleep(0.02)
        if self._thread.is_alive():
            self._thread.join(timeout=max(0.1, deadline - time.time()))
        if self._writer is not None:
            self._writer.release()
        if self._sidecar is not None:
            self._sidecar.flush()
            self._sidecar.close()
