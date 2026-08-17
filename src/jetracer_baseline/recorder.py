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

try:
    import Queue as queue  # Python 2 (khong dung nhung phong khi)
except ImportError:
    import queue

import cv2


class FrameRecorder(object):
    def __init__(self, path, fourcc='MJPG', fps=30.0, queue_size=30):
        self.path = path
        self._fourcc = fourcc
        self._fps = float(fps)
        self._queue = queue.Queue(maxsize=queue_size)
        self._writer = None
        self._sidecar = None
        self._video_index = 0
        self._last_frame_id = None
        self.n_written = 0
        self.n_dropped = 0
        self._stopped = False
        self._thread = threading.Thread(target=self._loop)
        self._thread.daemon = True
        self._thread.start()

    def submit(self, frame, frame_id):
        """Khong bao gio block. Bo qua neu frame_id trung lan truoc."""
        if frame_id == self._last_frame_id:
            return
        self._last_frame_id = frame_id
        try:
            self._queue.put_nowait((frame, frame_id, time.time()))
        except queue.Full:
            self.n_dropped += 1

    def _open_writer(self, frame):
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*self._fourcc)
        self._writer = cv2.VideoWriter(self.path, fourcc, self._fps, (w, h))
        sidecar_path = os.path.splitext(self.path)[0] + '.sidecar.csv'
        self._sidecar = io.open(sidecar_path, 'w', encoding='utf-8')
        self._sidecar.write(u'video_index,frame_id,timestamp\n')

    def _loop(self):
        while True:
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self._stopped:
                    return
                continue
            frame, frame_id, ts = item
            if self._writer is None:
                self._open_writer(frame)
            self._writer.write(frame)
            self._sidecar.write(u'%d,%d,%.3f\n' % (
                self._video_index, frame_id, ts))
            self._video_index += 1
            self.n_written += 1
            self._queue.task_done()

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
            self._sidecar.close()
