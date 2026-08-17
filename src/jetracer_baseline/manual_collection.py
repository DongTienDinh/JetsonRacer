# -*- coding: utf-8 -*-
"""Giao dien Jupyter de lai xe bang gamepad va thu dataset dong bo.

Module nay co hai phan tach biet:

* ``DatasetSessionWriter`` ghi anh goc + CSV. Phan nay khong phu thuoc
  Jupyter va co the test tren laptop.
* ``ManualDriveCollector`` tao ipywidgets Controller, xem truoc camera, gui
  lenh toi NvidiaRacecar va dieu khien viec ghi du lieu.

An toan:

* Mo camera khong tu dong khoi tao motor.
* Phai bam ARM moi gui lenh xuong xe.
* Mac dinh phai GIU nut LB (button 4) thi throttle moi khac 0.
* DISARM / DUNG KHAN CAP luon gui steering=0, throttle=0.

Python 3.6 compatible (Jetson Nano / JetPack 4.x).
"""

from __future__ import print_function

import atexit
import csv
import io
import json
import os
import re
import threading
import time

import cv2

from jetracer_baseline.camera import (LatestFrameGrabber, build_source,
                                      format_camera_environment)
from jetracer_baseline.config import load_config
from jetracer_baseline.control.driver import build_driver


CSV_FIELDS = [
    'sample_id',
    'image_file',
    'camera_frame_id',
    'timestamp_unix',
    'timestamp_monotonic',
    'steering_raw',
    'throttle_raw',
    'steering_cmd',
    'throttle_cmd',
    'deadman_pressed',
    'controller_connected',
]


def _clip(value, low, high):
    return max(low, min(high, float(value)))


def apply_deadzone(value, deadzone):
    """Bo vung chet nhung van anh xa phan con lai ve day [-1, 1]."""
    value = _clip(value, -1.0, 1.0)
    deadzone = _clip(deadzone, 0.0, 0.95)
    magnitude = abs(value)
    if magnitude <= deadzone:
        return 0.0
    shaped = (magnitude - deadzone) / (1.0 - deadzone)
    return shaped if value > 0 else -shaped


def _safe_session_name(value):
    value = (value or 'session').strip().lower()
    value = re.sub(r'[^a-z0-9_-]+', '_', value)
    value = value.strip('_')
    return value or 'session'


class _PacedFrameSource(object):
    """Gioi han nguon video ve FPS that de giao dien replay khong chay vun vut."""

    def __init__(self, source, fps):
        self.source = source
        self.period = 1.0 / max(1.0, float(fps))
        self._next_time = None

    def read(self):
        now = time.monotonic()
        if self._next_time is None:
            self._next_time = now
        delay = self._next_time - now
        if delay > 0:
            time.sleep(delay)
        ok, frame = self.source.read()
        self._next_time = max(self._next_time + self.period, time.monotonic())
        return ok, frame

    def release(self):
        self.source.release()


class DatasetSessionWriter(object):
    """Ghi mot session theo cau truc images/ + labels.csv + metadata.json."""

    def __init__(self, out_root, session, metadata=None, jpeg_quality=95,
                 flush_every=10):
        stamp = time.strftime('%Y%m%d_%H%M%S')
        base_name = '%s_%s' % (_safe_session_name(session), stamp)
        self.session_dir = os.path.join(out_root, base_name)
        suffix = 1
        while os.path.exists(self.session_dir):
            self.session_dir = os.path.join(
                out_root, '%s_%02d' % (base_name, suffix))
            suffix += 1

        self.images_dir = os.path.join(self.session_dir, 'images')
        os.makedirs(self.images_dir)
        self.csv_path = os.path.join(self.session_dir, 'labels.csv')
        self.metadata_path = os.path.join(self.session_dir, 'metadata.json')
        self._fh = io.open(self.csv_path, 'w', newline='', encoding='utf-8')
        self._csv = csv.DictWriter(self._fh, fieldnames=CSV_FIELDS)
        self._csv.writeheader()
        self._jpeg_quality = int(_clip(jpeg_quality, 50, 100))
        self._flush_every = max(1, int(flush_every))
        self._count = 0
        self._closed = False
        self._shape = None
        self._metadata = dict(metadata or {})
        self._metadata['session_name'] = base_name
        self._metadata['created_unix'] = time.time()
        self._write_metadata(completed=False)

    @property
    def count(self):
        return self._count

    def _write_metadata(self, completed):
        data = dict(self._metadata)
        data['completed'] = bool(completed)
        data['samples'] = self._count
        if self._shape is not None:
            data['frame_height'] = int(self._shape[0])
            data['frame_width'] = int(self._shape[1])
            data['frame_channels'] = int(self._shape[2]) if len(self._shape) > 2 else 1
        with io.open(self.metadata_path, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, sort_keys=True)

    def write(self, frame, camera_frame_id, timestamp_unix,
              timestamp_monotonic, steering_raw, throttle_raw,
              steering_cmd, throttle_cmd, deadman_pressed,
              controller_connected):
        if self._closed:
            raise RuntimeError('Session writer da dong')
        if frame is None or not hasattr(frame, 'shape'):
            raise ValueError('Frame khong hop le')

        sample_id = self._count
        image_file = 'frame_%06d.jpg' % sample_id
        image_path = os.path.join(self.images_dir, image_file)
        params = [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
        if not cv2.imwrite(image_path, frame, params):
            raise IOError('Khong ghi duoc anh: ' + image_path)

        self._shape = frame.shape
        self._csv.writerow({
            'sample_id': sample_id,
            'image_file': os.path.join('images', image_file).replace('\\', '/'),
            'camera_frame_id': int(camera_frame_id),
            'timestamp_unix': '%.6f' % float(timestamp_unix),
            'timestamp_monotonic': '%.6f' % float(timestamp_monotonic),
            'steering_raw': '%.6f' % float(steering_raw),
            'throttle_raw': '%.6f' % float(throttle_raw),
            'steering_cmd': '%.6f' % float(steering_cmd),
            'throttle_cmd': '%.6f' % float(throttle_cmd),
            'deadman_pressed': int(bool(deadman_pressed)),
            'controller_connected': int(bool(controller_connected)),
        })
        self._count += 1
        if self._count % self._flush_every == 0:
            self._fh.flush()
        return image_path

    def close(self):
        if self._closed:
            return
        self._fh.flush()
        self._fh.close()
        self._closed = True
        self._write_metadata(completed=True)


class ManualDriveCollector(object):
    """UI Jupyter: preview camera, gamepad, NvidiaRacecar va recorder."""

    def __init__(self, config_path='configs/default.yaml', out_root='data/driving',
                 source_kind='csi', video_path=None, driver_kind='nvidia',
                 controller_index=0):
        try:
            import ipywidgets.widgets as widgets
        except ImportError:
            raise ImportError(
                'Can ipywidgets trong moi truong Jupyter. Kiem tra bang: '
                'python3 -c "import ipywidgets; print(ipywidgets.__version__)"')

        self.widgets = widgets
        self.config_path = config_path
        self.cfg = load_config(config_path)
        self.out_root = out_root
        self.source_kind = source_kind
        self.video_path = video_path
        self.driver_kind = driver_kind

        self.controller = widgets.Controller(index=controller_index)
        self.preview = widgets.Image(format='jpeg', width=480)
        self.output = widgets.Output(
            layout=widgets.Layout(border='1px solid #ddd', height='150px',
                                  overflow_y='auto'))
        self.status = widgets.HTML(value='<b>Trang thai:</b> chua mo camera')
        self.command_view = widgets.HTML(value=self._command_html())
        self.controller_view = widgets.HTML(value='<b>Tay cam:</b> dang cho...')
        self.driver_view = widgets.HTML(value=self._driver_html())

        self.session_name = widgets.Text(
            value='bai1_sang', description='Session:',
            layout=widgets.Layout(width='320px'))
        self.steering_axis = widgets.BoundedIntText(
            value=2, min=0, max=15, description='Truc lai:')
        self.throttle_axis = widgets.BoundedIntText(
            value=1, min=0, max=15, description='Truc ga:')
        self.invert_steering = widgets.Checkbox(
            value=False, description='Dao truc lai')
        self.invert_throttle = widgets.Checkbox(
            value=True, description='Dao truc ga')
        self.deadzone = widgets.FloatSlider(
            value=0.06, min=0.0, max=0.25, step=0.01,
            description='Deadzone:', readout_format='.2f')
        self.max_throttle = widgets.FloatSlider(
            value=0.20, min=0.05, max=0.40, step=0.01,
            description='Ga toi da:', readout_format='.2f')
        self.save_hz = widgets.FloatSlider(
            value=5.0, min=1.0, max=20.0, step=1.0,
            description='Luu FPS:', readout_format='.0f')
        self.use_deadman = widgets.Checkbox(
            value=True, description='Bat buoc giu dead-man')
        self.deadman_button = widgets.BoundedIntText(
            value=4, min=0, max=31, description='Nut dead-man:')
        self.wheels_lifted = widgets.Checkbox(
            value=False, description='BANH XE DA KE KHOI MAT DAT',
            style={'description_width': 'initial'})
        self.test_steering = widgets.FloatSlider(
            value=0.45, min=0.10, max=0.75, step=0.05,
            description='Test lai:', readout_format='.2f')
        self.test_throttle = widgets.FloatSlider(
            value=0.18, min=0.10, max=0.30, step=0.01,
            description='Test ga:', readout_format='.2f')

        self.btn_probe = widgets.Button(
            description='0. KIEM TRA TAY CAM', button_style='info',
            layout=widgets.Layout(width='210px', height='38px'))
        self.btn_camera = widgets.Button(
            description='1. MO CAMERA', button_style='info',
            layout=widgets.Layout(width='180px', height='38px'))
        self.btn_arm = widgets.Button(
            description='2. ARM TAY CAM', button_style='warning',
            layout=widgets.Layout(width='180px', height='38px'))
        self.btn_record = widgets.Button(
            description='3. BAT DAU GHI', button_style='success',
            layout=widgets.Layout(width='180px', height='38px'))
        self.btn_stop_record = widgets.Button(
            description='DUNG GHI',
            layout=widgets.Layout(width='130px', height='38px'))
        self.btn_emergency = widgets.Button(
            description='DUNG KHAN CAP', button_style='danger',
            layout=widgets.Layout(width='180px', height='46px'))
        self.btn_close = widgets.Button(
            description='DONG CAMERA',
            layout=widgets.Layout(width='150px', height='38px'))
        self.btn_disarm = widgets.Button(
            description='DISARM', button_style='warning',
            layout=widgets.Layout(width='120px', height='38px'))
        self.btn_test_steering = widgets.Button(
            description='TEST SERVO', button_style='warning',
            layout=widgets.Layout(width='150px', height='38px'))
        self.btn_test_motor = widgets.Button(
            description='TEST MOTOR 0.5S', button_style='danger',
            layout=widgets.Layout(width='170px', height='38px'))

        self.btn_probe.on_click(self._on_probe_controller)
        self.btn_camera.on_click(self._on_open_camera)
        self.btn_arm.on_click(self._on_arm)
        self.btn_record.on_click(self._on_start_recording)
        self.btn_stop_record.on_click(self._on_stop_recording)
        self.btn_emergency.on_click(self._on_emergency)
        self.btn_close.on_click(self._on_close)
        self.btn_disarm.on_click(self._on_disarm)
        self.btn_test_steering.on_click(self._on_test_steering)
        self.btn_test_motor.on_click(self._on_test_motor)

        self._grabber = None
        self._driver = None
        self._writer = None
        self._camera_thread = None
        self._control_thread = None
        self._stop_event = threading.Event()
        self._armed = False
        self._recording = False
        self._state_lock = threading.Lock()
        self._steering_raw = 0.0
        self._throttle_raw = 0.0
        self._steering_cmd = 0.0
        self._throttle_cmd = 0.0
        self._deadman_pressed = False
        self._camera_fps = 0.0
        self._last_session_dir = None
        self._closed = False
        self._actuator_test_running = False
        self._actuator_thread = None
        self._actuator_stop_event = threading.Event()

        atexit.register(self.close)

    def _command_html(self):
        return (
            '<b>Lenh gui xuong xe:</b> steering=%+.3f &nbsp; throttle=%+.3f '
            '&nbsp; dead-man=%s &nbsp; armed=%s' % (
                getattr(self, '_steering_cmd', 0.0),
                getattr(self, '_throttle_cmd', 0.0),
                'ON' if getattr(self, '_deadman_pressed', False) else 'OFF',
                'YES' if getattr(self, '_armed', False) else 'NO'))

    def _driver_html(self):
        driver = getattr(self, '_driver', None)
        driver_name = type(driver).__name__ if driver is not None else 'CHUA KET NOI'
        color = '#087f23' if driver is not None and self.driver_kind == 'nvidia' \
            else '#b26a00'
        return (
            '<b>Driver:</b> backend=<span style="color:%s">%s</span> &nbsp; '
            'object=%s' % (color, self.driver_kind, driver_name))

    def _controller_html(self):
        connected = bool(getattr(self.controller, 'connected', False))
        name = getattr(self.controller, 'name', '') or '(khong co ten)'
        axes = []
        for index, axis in enumerate(self.controller.axes):
            try:
                axis_value = float(axis.value)
            except (TypeError, ValueError):
                axis_value = 0.0
            axes.append('%d:%+.2f' % (index, axis_value))
        pressed = []
        for index, button in enumerate(self.controller.buttons):
            try:
                button_value = float(getattr(button, 'value', 0.0))
            except (TypeError, ValueError):
                button_value = 0.0
            is_pressed = bool(getattr(button, 'pressed', False)) or \
                button_value > 0.5
            if is_pressed:
                pressed.append(str(index))
        color = '#087f23' if connected else '#b00020'
        return (
            '<b>Tay cam:</b> <span style="color:%s">connected=%s</span> '
            '&nbsp; name=%s<br>'
            '<b>Axes live:</b> [%s]<br>'
            '<b>Buttons dang bam:</b> [%s]' % (
                color, connected, name, ', '.join(axes), ', '.join(pressed)))

    def _log(self, message):
        with self.output:
            print('[%s] %s' % (time.strftime('%H:%M:%S'), message))

    def _set_status(self, message):
        self.status.value = '<b>Trang thai:</b> ' + message

    def _controller_ready(self):
        max_axis = max(int(self.steering_axis.value), int(self.throttle_axis.value))
        return bool(getattr(self.controller, 'connected', False)) and \
            len(self.controller.axes) > max_axis

    def _ensure_control_thread(self):
        if self._control_thread is not None and self._control_thread.is_alive():
            return
        self._stop_event.clear()
        self._control_thread = threading.Thread(target=self._control_loop)
        self._control_thread.daemon = True
        self._control_thread.start()

    def _ensure_driver(self):
        if self._driver is None:
            self._driver = build_driver(self.driver_kind, self.cfg)
            self._driver.stop()
            self.driver_view.value = self._driver_html()
            self._log('Driver da ket noi: backend=%s, object=%s' % (
                self.driver_kind, type(self._driver).__name__))
        return self._driver

    def _deadman_value(self):
        if not self.use_deadman.value:
            return True
        index = int(self.deadman_button.value)
        if len(self.controller.buttons) <= index:
            return False
        button = self.controller.buttons[index]
        if hasattr(button, 'pressed'):
            return bool(button.pressed)
        return float(getattr(button, 'value', 0.0)) > 0.5

    def _read_controller_command(self):
        steer_index = int(self.steering_axis.value)
        throttle_index = int(self.throttle_axis.value)
        steer_raw = float(self.controller.axes[steer_index].value)
        throttle_raw = float(self.controller.axes[throttle_index].value)

        steer = apply_deadzone(steer_raw, self.deadzone.value)
        throttle = apply_deadzone(throttle_raw, self.deadzone.value)
        if self.invert_steering.value:
            steer = -steer
        if self.invert_throttle.value:
            throttle = -throttle
        throttle *= float(self.max_throttle.value)

        deadman = self._deadman_value()
        if not deadman:
            throttle = 0.0
        return steer_raw, throttle_raw, steer, throttle, deadman

    def _zero_command(self):
        with self._state_lock:
            self._steering_raw = 0.0
            self._throttle_raw = 0.0
            self._steering_cmd = 0.0
            self._throttle_cmd = 0.0
            self._deadman_pressed = False
        if self._driver is not None:
            try:
                self._driver.stop()
            except Exception as exc:
                self._log('Khong gui duoc lenh stop: %s' % exc)
        self.command_view.value = self._command_html()

    def _control_loop(self):
        last_diag = 0.0
        while not self._stop_event.is_set():
            now = time.monotonic()
            if (now - last_diag) >= 0.10:
                last_diag = now
                try:
                    self.controller_view.value = self._controller_html()
                    self.driver_view.value = self._driver_html()
                    self.command_view.value = self._command_html()
                except Exception:
                    pass
            if not self._armed:
                time.sleep(0.05)
                continue
            if not self._controller_ready():
                self._log('Mat ket noi tay cam -> tu dong DISARM va throttle=0')
                self._armed = False
                self._zero_command()
                self._set_status('MAT TAY CAM - da dung xe')
                continue
            try:
                values = self._read_controller_command()
                with self._state_lock:
                    self._steering_raw = values[0]
                    self._throttle_raw = values[1]
                    self._steering_cmd = values[2]
                    self._throttle_cmd = values[3]
                    self._deadman_pressed = values[4]
                self._driver.set(values[2], values[3])
            except Exception as exc:
                self._log('Loi dieu khien -> DISARM: %s' % exc)
                self._armed = False
                self._zero_command()
            time.sleep(1.0 / 30.0)

    def _camera_loop(self, grabber):
        last_frame_id = -1
        last_preview = 0.0
        last_save = 0.0
        fps_t0 = time.monotonic()
        fps_frames = 0

        while not self._stop_event.is_set():
            frame, frame_id = grabber.read()
            if frame is None or frame_id == last_frame_id:
                if getattr(grabber, 'eof', False) or grabber.error is not None:
                    if grabber.error is not None:
                        self._log('CAMERA DUNG DO LOI: %s' % grabber.error)
                    else:
                        self._log('Nguon camera/video da ket thuc.')
                    break
                time.sleep(0.005)
                continue

            last_frame_id = frame_id
            now_mono = time.monotonic()
            now_unix = time.time()
            fps_frames += 1
            elapsed = now_mono - fps_t0
            if elapsed >= 1.0:
                self._camera_fps = fps_frames / elapsed
                fps_t0 = now_mono
                fps_frames = 0

            if self._recording and self._writer is not None:
                interval = 1.0 / max(1.0, float(self.save_hz.value))
                if (now_mono - last_save) >= interval:
                    last_save = now_mono
                    try:
                        with self._state_lock:
                            state = (
                                self._steering_raw, self._throttle_raw,
                                self._steering_cmd, self._throttle_cmd,
                                self._deadman_pressed)
                        self._writer.write(
                            frame, frame_id, now_unix, now_mono,
                            state[0], state[1], state[2], state[3], state[4],
                            getattr(self.controller, 'connected', False))
                    except Exception as exc:
                        self._log('LOI GHI DATA -> dung ghi: %s' % exc)
                        self._stop_recording()

            if (now_mono - last_preview) >= 0.10:
                last_preview = now_mono
                preview = frame.copy()
                color = (0, 0, 255) if self._recording else (0, 200, 0)
                label = 'REC' if self._recording else 'PREVIEW'
                cv2.putText(preview, '%s  FPS %.1f' % (label, self._camera_fps),
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                ok, encoded = cv2.imencode(
                    '.jpg', preview, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ok:
                    self.preview.value = encoded.tobytes()

        if self._armed:
            self._armed = False
            self._zero_command()
        if self._recording:
            self._log('Camera dung -> tu dong dong session dang ghi.')
            self._stop_recording()
        try:
            grabber.stop()
        except Exception as exc:
            self._log('Loi release camera: %s' % exc)
        if self._grabber is grabber:
            self._grabber = None
        if grabber.error is not None:
            self._set_status('CAMERA LOI - xem log; co the bam MO CAMERA de thu lai')
        else:
            self._set_status('camera da dung; co the bam MO CAMERA de mo lai')

    def _on_open_camera(self, _button=None):
        if self._grabber is not None:
            self._log('Camera dang mo.')
            return
        source = None
        grabber = None
        try:
            self._log('Dang mo camera kind=%s, %s' % (
                self.source_kind, format_camera_environment()))
            source = build_source(
                self.cfg, self.source_kind, video_path=self.video_path)
            if self.source_kind == 'video':
                video_fps = source.cap.get(cv2.CAP_PROP_FPS)
                if video_fps <= 0:
                    video_fps = self.cfg.get('camera.fps', 20)
                source = _PacedFrameSource(source, video_fps)
            grabber = LatestFrameGrabber(source).start()
            frame, frame_id = grabber.read()
            if frame is None:
                error = grabber.error
                grabber.stop()
                grabber = None
                source = None
                if error is not None:
                    raise IOError(str(error))
                raise IOError('Mo duoc nguon nhung khong nhan frame trong 5 giay')
            self._grabber = grabber
            self._stop_event.clear()
            self._camera_thread = threading.Thread(
                target=self._camera_loop, args=(grabber,))
            self._camera_thread.daemon = True
            self._camera_thread.start()
            self._ensure_control_thread()
            h, w = frame.shape[:2]
            backend = getattr(source, 'backend', self.source_kind)
            self._set_status(
                'camera OK %dx%d backend=%s; chua ARM motor' %
                (w, h, backend))
            self._log('Camera OK: %dx%d, frame_id=%d, backend=%s' %
                      (w, h, frame_id, backend))
            for note in getattr(source, 'startup_notes', []):
                self._log('Camera note: %s' % note)
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
            self._grabber = None
            self._set_status('LOI CAMERA')
            self._log('Khong mo duoc camera: %s' % exc)
            self._log(
                'Dong cac notebook/kernel camera khac. Neu dung CSI, thu trong '
                'terminal: sudo systemctl restart nvargus-daemon')

    def _on_arm(self, _button=None):
        if self._grabber is None:
            self._log('Hay MO CAMERA va kiem tra hinh truoc khi ARM.')
            return
        try:
            steer_raw = float(
                self.controller.axes[int(self.steering_axis.value)].value)
            throttle_raw = float(
                self.controller.axes[int(self.throttle_axis.value)].value)
        except Exception as exc:
            self._log('Khong doc duoc truc tay cam: %s' % exc)
            return
        if abs(steer_raw) > 0.20 or abs(throttle_raw) > 0.20:
            self._log(
                'TU CHOI ARM: can gat chua ve gan 0 '
                '(steer=%+.2f, throttle=%+.2f).' % (steer_raw, throttle_raw))
            self._set_status('Hay tha hai can gat ve vi tri giua roi ARM lai')
            return
        if not self._controller_ready():
            self._log(
                'Tay cam chua san sang. Bam/xoay cac can, sau do kiem tra '
                'connected va so truc.')
            return
        if self.use_deadman.value and \
                len(self.controller.buttons) <= int(self.deadman_button.value):
            self._log('Tay cam khong co button %d. Chon lai nut dead-man.' %
                      int(self.deadman_button.value))
            return
        try:
            self._ensure_driver()
            self._armed = True
            if self.use_deadman.value:
                requirement = 'giu button %d de co ga' % int(
                    self.deadman_button.value)
            else:
                requirement = 'dead-man DANG TAT'
            self._set_status('DA ARM; %s' % requirement)
            self._log(
                'ARM OK: driver=%s, steering=axis[%d], throttle=axis[%d], '
                'max=%.2f, %s' % (
                    type(self._driver).__name__,
                    int(self.steering_axis.value), int(self.throttle_axis.value),
                    float(self.max_throttle.value), requirement))
        except Exception as exc:
            self._armed = False
            self._zero_command()
            self._set_status('LOI DRIVER')
            self._log('Khong ARM duoc driver %s: %s' % (self.driver_kind, exc))

    def _on_probe_controller(self, _button=None):
        self._ensure_control_thread()
        connected = bool(getattr(self.controller, 'connected', False))
        n_axes = len(self.controller.axes)
        n_buttons = len(self.controller.buttons)
        self._log('Tay cam: connected=%s, axes=%d, buttons=%d, name=%s' % (
            connected, n_axes, n_buttons,
            getattr(self.controller, 'name', '') or '(khong co ten)'))
        if not connected:
            self._set_status('TAY CAM CHUA KET NOI')
        elif n_axes <= max(int(self.steering_axis.value),
                           int(self.throttle_axis.value)):
            self._set_status('SAI SO TRUC - xem bang Axes live')
        else:
            self._set_status('TAY CAM OK - xem Axes live va Buttons dang bam')

    def _require_lifted_driver(self):
        if not self.wheels_lifted.value:
            self._log(
                'TU CHOI TEST: chua xac nhan BANH XE DA KE KHOI MAT DAT.')
            self._set_status('Phai ke banh xe truoc khi test actuator')
            return None
        if self._actuator_test_running:
            self._log('Mot bai test actuator dang chay.')
            return None
        if self._recording:
            self._log('TU CHOI TEST: hay DUNG GHI truoc khi test actuator.')
            return None
        try:
            driver = self._ensure_driver()
        except Exception as exc:
            self._set_status('LOI DRIVER')
            self._log('Khong ket noi duoc driver %s: %s' % (
                self.driver_kind, exc))
            return None
        if self.driver_kind != 'nvidia':
            self._log(
                'CANH BAO: backend=%s chi mo phong, xe se khong chuyen dong.' %
                self.driver_kind)
        self._armed = False
        self._zero_command()
        return driver

    def _run_actuator_test(self, kind):
        driver = self._require_lifted_driver()
        if driver is None:
            return
        self._actuator_test_running = True
        self._actuator_stop_event.clear()

        def worker():
            try:
                if kind == 'steering':
                    value = float(self.test_steering.value)
                    self._set_status(
                        'DANG TEST SERVO - trai/giua/phai/giua')
                    for steering in (-value, 0.0, value, 0.0):
                        if self._actuator_stop_event.is_set():
                            break
                        driver.set(steering, 0.0)
                        if self._actuator_stop_event.wait(0.7):
                            break
                    self._log('TEST SERVO xong: bien do +/-%0.2f' % value)
                else:
                    value = float(self.test_throttle.value)
                    self._set_status('DANG TEST MOTOR 0.5 GIAY')
                    driver.set(0.0, value)
                    self._actuator_stop_event.wait(0.5)
                    self._log('TEST MOTOR xong: throttle=+%.2f trong 0.5 s' % value)
            except Exception as exc:
                self._log('LOI TEST ACTUATOR: %s' % exc)
            finally:
                try:
                    driver.stop()
                except Exception:
                    pass
                self._actuator_test_running = False
                self._zero_command()
                self._set_status('TEST XONG; xe dang DISARM')

        self._actuator_thread = threading.Thread(target=worker)
        self._actuator_thread.daemon = True
        self._actuator_thread.start()

    def _on_test_steering(self, _button=None):
        self._run_actuator_test('steering')

    def _on_test_motor(self, _button=None):
        self._run_actuator_test('motor')

    def _metadata(self):
        return {
            'purpose': 'manual_driving_collection',
            'config_path': self.config_path,
            'source_kind': self.source_kind,
            'video_path': self.video_path,
            'driver_kind': self.driver_kind,
            'camera_config': {
                'width': self.cfg.get('camera.width'),
                'height': self.cfg.get('camera.height'),
                'fps': self.cfg.get('camera.fps'),
            },
            'controller_mapping': {
                'steering_axis': int(self.steering_axis.value),
                'throttle_axis': int(self.throttle_axis.value),
                'invert_steering': bool(self.invert_steering.value),
                'invert_throttle': bool(self.invert_throttle.value),
                'deadzone': float(self.deadzone.value),
                'max_throttle': float(self.max_throttle.value),
                'use_deadman': bool(self.use_deadman.value),
                'deadman_button': int(self.deadman_button.value),
            },
            'save_hz_requested': float(self.save_hz.value),
        }

    def _on_start_recording(self, _button=None):
        if self._grabber is None:
            self._log('Chua co camera.')
            return
        if not self._armed:
            self._log('Chua ARM tay cam/driver. Khong ghi nhan lai xe khong dong bo.')
            return
        if self._recording:
            self._log('Dang ghi roi.')
            return
        try:
            self._writer = DatasetSessionWriter(
                self.out_root, self.session_name.value,
                metadata=self._metadata())
            self._last_session_dir = self._writer.session_dir
            self._recording = True
            self._set_status('DANG GHI -> %s' % self._writer.session_dir)
            self._log('BAT DAU GHI: %s' % self._writer.session_dir)
        except Exception as exc:
            self._writer = None
            self._recording = False
            self._log('Khong tao duoc session: %s' % exc)

    def _stop_recording(self):
        self._recording = False
        writer = self._writer
        self._writer = None
        if writer is not None:
            writer.close()
            self._last_session_dir = writer.session_dir
            self._log('DUNG GHI: %d anh -> %s' % (writer.count, writer.session_dir))
        if self._armed:
            self._set_status('da dung ghi; xe van ARM')

    def _on_stop_recording(self, _button=None):
        self._stop_recording()

    def _on_emergency(self, _button=None):
        self._actuator_stop_event.set()
        self._stop_recording()
        self._armed = False
        self._zero_command()
        self._set_status('DUNG KHAN CAP - motor=0, steering=0')
        self._log('DUNG KHAN CAP: da DISARM xe.')

    def _on_disarm(self, _button=None):
        self._actuator_stop_event.set()
        self._stop_recording()
        self._armed = False
        self._zero_command()
        self._set_status('DA DISARM - motor=0, steering=0')
        self._log('Da DISARM xe.')

    def _on_close(self, _button=None):
        self.close()

    def widget(self):
        w = self.widgets
        warning = w.HTML(
            '<b style="color:#b00020">AN TOAN:</b> lan dau ke banh xe khoi mat '
            'dat. Mac dinh phai GIU LB/button 4 thi xe moi co ga. Nut do khan '
            'cap vat ly/nguoi bat xe van bat buoc.')
        settings1 = w.HBox([
            self.session_name, self.steering_axis, self.throttle_axis])
        settings2 = w.HBox([
            self.invert_steering, self.invert_throttle, self.deadzone])
        settings3 = w.HBox([
            self.max_throttle, self.save_hz,
            self.use_deadman, self.deadman_button])
        hardware_test = w.VBox([
            w.HTML('<b>Test phan cung doc lap (chi khi xe da ke):</b>'),
            w.HBox([self.wheels_lifted, self.test_steering,
                    self.test_throttle]),
            w.HBox([self.btn_test_steering, self.btn_test_motor]),
        ])
        buttons = w.HBox([
            self.btn_probe, self.btn_camera, self.btn_arm, self.btn_record])
        stop_buttons = w.HBox([
            self.btn_stop_record, self.btn_disarm, self.btn_close])
        return w.VBox([
            warning,
            w.HTML('<b>Tay cam (bam nut/xoay can de trinh duyet nhan):</b>'),
            self.controller,
            self.controller_view,
            settings1,
            settings2,
            settings3,
            hardware_test,
            buttons,
            stop_buttons,
            self.btn_emergency,
            self.status,
            self.driver_view,
            self.command_view,
            self.preview,
            self.output,
        ])

    def show(self):
        try:
            from IPython.display import display
        except ImportError:
            raise RuntimeError('Ham show() phai chay trong Jupyter/IPython')
        display(self.widget())
        self._ensure_control_thread()
        self._log(
            'Thu tu: KIEM TRA TAY CAM -> test actuator khi xe da ke -> '
            'MO CAMERA -> ARM -> giu dead-man -> GHI.')
        return self

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._actuator_stop_event.set()
        self._stop_recording()
        self._armed = False
        self._zero_command()
        self._stop_event.set()

        if self._camera_thread is not None and self._camera_thread.is_alive():
            self._camera_thread.join(timeout=1.0)
        if self._control_thread is not None and self._control_thread.is_alive():
            self._control_thread.join(timeout=1.0)
        if self._actuator_thread is not None and self._actuator_thread.is_alive():
            self._actuator_thread.join(timeout=1.0)
        if self._grabber is not None:
            try:
                self._grabber.stop()
            except Exception as exc:
                self._log('Loi dong camera: %s' % exc)
            self._grabber = None
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception as exc:
                self._log('Loi dong driver: %s' % exc)
            self._driver = None
        self._set_status('DA DONG camera va driver')
        self._log('Da dong collector an toan.')
        for button in (self.btn_camera, self.btn_arm, self.btn_record,
                       self.btn_stop_record, self.btn_emergency, self.btn_close,
                       self.btn_probe, self.btn_disarm,
                       self.btn_test_steering, self.btn_test_motor):
            button.disabled = True

    @property
    def last_session_dir(self):
        return self._last_session_dir


def launch_manual_collection(config_path='configs/default.yaml',
                             out_root='data/driving', source_kind='csi',
                             video_path=None, driver_kind='nvidia',
                             controller_index=0):
    collector = ManualDriveCollector(
        config_path=config_path,
        out_root=out_root,
        source_kind=source_kind,
        video_path=video_path,
        driver_kind=driver_kind,
        controller_index=controller_index)
    collector.show()
    return collector
