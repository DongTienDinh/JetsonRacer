# -*- coding: utf-8 -*-
"""Smoke test chay duoc tren laptop, khong can xe, khong can model.

    python -m pytest tests/ -q      (hoac)      python tests/test_smoke.py

Muc dich: bat loi tich hop TRUOC khi mang code len xe. Gio chay tren xe la tai
nguyen khan hiem nhat (5 xe / 10 doi) - khong duoc dung de debug loi cu phap.
"""

import os
import csv
import shutil
import sys
import tempfile
import threading
import time
import types

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from jetracer_baseline import fsm as fsm_mod                     # noqa: E402
from jetracer_baseline.camera import (                          # noqa: E402
    CSISource, LatestFrameGrabber, SyntheticSource,
    gstreamer_pipeline)
from jetracer_baseline.config import load_config                 # noqa: E402
from jetracer_baseline.control.pid import PID                    # noqa: E402
from jetracer_baseline.control import driver as driver_mod       # noqa: E402
from jetracer_baseline.perception.lane import LaneDetector       # noqa: E402
from jetracer_baseline.perception.shading import (               # noqa: E402
    ShadingCorrector, fit_coefficients, measure_ratio_profile)
from jetracer_baseline.perception.signs import SignTracker       # noqa: E402
from jetracer_baseline.pipeline import Runner                    # noqa: E402
from jetracer_baseline.tuning_ui import (                        # noqa: E402
    LaneTuningEngine, RollingStats, _dict_diff)
from jetracer_baseline.manual_collection import (                # noqa: E402
    DatasetSessionWriter, ManualDriveCollector, apply_deadzone,
    shape_steering, shape_throttle, slew_towards)

CONFIG = os.path.join(ROOT, 'configs', 'default.yaml')


def _cfg(**overrides):
    cfg = load_config(CONFIG)
    for key, value in overrides.items():
        cfg.set(key.replace('__', '.'), value)
    return cfg


def test_config_loads():
    cfg = load_config(CONFIG)
    assert cfg.get('control.v_max') > 0
    assert cfg.get('pipeline.control_hz') >= 20
    assert cfg.get('control.driver.implementation') == 'waveshare_single_pca'
    assert cfg.get('control.driver.i2c_address') == 0x40
    assert cfg.get('control.driver.steering_channel') == 0
    assert cfg.get('control.driver.throttle_channel') == 1
    assert cfg.get('control.driver.command_timeout_s') > 0
    assert cfg.get('control.driver.pwm_frequency') is None
    assert abs(cfg.get('control.driver.steering_gain')) <= 0.65
    assert cfg.get('control.driver.pulse_width_range') == [750, 2250]
    assert cfg.get('control.driver.steering_output_min') == -0.40
    assert cfg.get('control.driver.steering_output_max') == 0.40
    assert 0 < cfg.get('manual.min_throttle') < cfg.get('manual.max_throttle')
    assert cfg.get('manual.test_throttle') >= cfg.get('manual.min_throttle')
    assert 0 < cfg.get('manual.max_steering') <= 0.60
    assert cfg.get('manual.steering_slew_rate') > 0
    assert 10 <= cfg.get('manual.video_fps') <= cfg.get('camera.fps')
    assert cfg.get('control.steer_max') <= 0.60
    assert cfg.get('fsm.turn_steer') <= cfg.get('control.steer_max')
    # Doc override phai ghi de dung, khong lam mat khoa khac
    cfg2 = load_config(CONFIG, [os.path.join(ROOT, 'configs', 'fast.yaml')])
    assert cfg2.get('control.v_max') > cfg.get('control.v_max')
    assert cfg2.get('lane.roi_top') == cfg.get('lane.roi_top')


def test_pid_sign_and_limit():
    pid = PID(kp=1.0, ki=0.0, kd=0.0, out_limit=0.5)
    assert pid.step(0.3, 0.033) > 0        # lech phai -> lai phai
    pid.reset()
    assert pid.step(-0.3, 0.033) < 0       # lech trai -> lai trai
    pid.reset()
    assert abs(pid.step(10.0, 0.033)) <= 0.5   # ton trong out_limit


def test_manual_collection_deadzone_and_writer():
    assert apply_deadzone(0.03, 0.06) == 0.0
    assert apply_deadzone(-0.03, 0.06) == 0.0
    assert 0.45 < apply_deadzone(0.50, 0.06) < 0.50
    assert shape_throttle(0.03, 0.06, 0.12, 0.30) == 0.0
    assert abs(shape_throttle(1.0, 0.06, 0.12, 0.30) - 0.30) < 1e-9
    assert -0.22 < shape_throttle(-0.50, 0.06, 0.12, 0.30) < -0.19
    assert shape_steering(0.03, 0.06, 0.85, 0.35) == 0.0
    assert abs(shape_steering(1.0, 0.06, 0.85, 0.35) - 0.85) < 1e-9
    assert 0.25 < shape_steering(0.50, 0.06, 0.85, 0.35) < 0.32
    assert abs(slew_towards(0.0, 1.0, 2.0, 0.1) - 0.2) < 1e-9

    temp_root = tempfile.mkdtemp(prefix='jetracer_collect_')
    writer = None
    try:
        writer = DatasetSessionWriter(
            temp_root, 'test sang', metadata={'purpose': 'smoke'})
        frame = np.full((24, 32, 3), 80, dtype=np.uint8)
        path = writer.write(
            frame, camera_frame_id=7,
            timestamp_unix=1000.25, timestamp_monotonic=20.5,
            steering_raw=0.4, throttle_raw=-0.6,
            steering_cmd=0.36, throttle_cmd=0.12,
            deadman_pressed=True, controller_connected=True)
        writer.close()
        assert os.path.exists(path)
        assert os.path.exists(writer.metadata_path)
        with open(writer.csv_path, 'r') as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert rows[0]['camera_frame_id'] == '7'
        assert rows[0]['deadman_pressed'] == '1'
        assert rows[0]['image_file'] == 'images/frame_000000.jpg'
    finally:
        if writer is not None:
            writer.close()
        shutil.rmtree(temp_root)


def test_csi_open_without_frame_falls_back_to_usb():
    """Bat loi Jetson hay gap: isOpened=True nhung nvargus khong tra frame."""
    import jetracer_baseline.camera as camera_mod

    frame = np.full((12, 16, 3), 123, dtype=np.uint8)

    class FakeCapture(object):
        def __init__(self, opened, reads):
            self.opened = opened
            self.reads = list(reads)
            self.released = False

        def isOpened(self):
            return self.opened

        def read(self):
            if not self.reads:
                return False, None
            return self.reads.pop(0)

        def set(self, _key, _value):
            return True

        def release(self):
            self.released = True

    csi_cap = FakeCapture(True, [(False, None), (False, None)])
    usb_cap = FakeCapture(True, [(True, frame)])
    original = camera_mod.cv2.VideoCapture

    def fake_video_capture(source, *args):
        if isinstance(source, str):
            return csi_cap
        return usb_cap

    camera_mod.cv2.VideoCapture = fake_video_capture
    source = None
    try:
        source = CSISource(
            640, 480, 30, allow_usb_fallback=True,
            startup_attempts=2, startup_delay_s=0.0,
            csi_open_attempts=1, csi_retry_delay_s=0.0)
        ok, result = source.read()
        assert ok
        assert result.shape == frame.shape
        assert source.backend == 'usb-v4l2-index-0'
        assert csi_cap.released
        assert source.startup_notes
    finally:
        if source is not None:
            source.release()
        camera_mod.cv2.VideoCapture = original


def test_csi_open_without_frame_reopens_argus_before_usb():
    """CaptureSession loi tam thoi: release va mo lai CSI truoc khi fallback."""
    import jetracer_baseline.camera as camera_mod

    frame = np.full((12, 16, 3), 77, dtype=np.uint8)

    class FakeCapture(object):
        def __init__(self, reads):
            self.reads = list(reads)
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            if not self.reads:
                return False, None
            return self.reads.pop(0)

        def release(self):
            self.released = True

    first = FakeCapture([(False, None)])
    second = FakeCapture([(True, frame)])
    captures = [first, second]
    original = camera_mod.cv2.VideoCapture

    def fake_video_capture(_source, *_args):
        return captures.pop(0)

    camera_mod.cv2.VideoCapture = fake_video_capture
    source = None
    try:
        source = CSISource(
            640, 480, 30, allow_usb_fallback=True,
            startup_attempts=1, startup_delay_s=0.0,
            csi_open_attempts=2, csi_retry_delay_s=0.0)
        ok, result = source.read()
        assert ok
        assert result.shape == frame.shape
        assert source.backend == 'csi-gstreamer'
        assert first.released
        assert any('mo lai lan 2/2' in note for note in source.startup_notes)
        assert not captures
    finally:
        if source is not None:
            source.release()
        camera_mod.cv2.VideoCapture = original


def test_jetson_gstreamer_pipeline_has_low_latency_caps():
    pipeline = gstreamer_pipeline(
        640, 480, 30, flip_method=2,
        capture_width=1280, capture_height=720, sensor_id=0)
    assert 'nvarguscamerasrc' in pipeline
    assert 'sensor-id=0' in pipeline
    assert 'memory:NVMM' in pipeline
    assert 'width=(int)1280, height=(int)720' in pipeline
    assert 'format=(string)NV12' in pipeline
    assert 'nvvidconv flip-method=2' in pipeline
    assert 'format=(string)BGR' in pipeline
    assert 'drop=true' in pipeline
    assert 'max-buffers=1' in pipeline
    assert 'sync=false' in pipeline


def test_nvidia_driver_finds_sibling_repo_without_install():
    """Jetson co ~/jetracer repo nhung chua setup.py install van phai import."""
    temp_root = tempfile.mkdtemp(prefix='nvidia_jetracer_repo_')
    package = os.path.join(temp_root, 'jetracer')
    old_env = os.environ.get('JETRACER_NVIDIA_ROOT')
    old_path = list(sys.path)
    old_package = sys.modules.get('jetracer')
    old_module = sys.modules.get('jetracer.nvidia_racecar')
    try:
        os.makedirs(package)
        with open(os.path.join(package, '__init__.py'), 'w') as fh:
            fh.write('')
        with open(os.path.join(package, 'nvidia_racecar.py'), 'w') as fh:
            fh.write('class NvidiaRacecar(object):\n    pass\n')
        os.environ['JETRACER_NVIDIA_ROOT'] = temp_root
        sys.modules.pop('jetracer.nvidia_racecar', None)
        sys.modules.pop('jetracer', None)

        cls, module_path = driver_mod._load_nvidia_racecar()
        assert cls.__name__ == 'NvidiaRacecar'
        assert os.path.abspath(temp_root) in os.path.abspath(module_path)
    finally:
        sys.path[:] = old_path
        sys.modules.pop('jetracer.nvidia_racecar', None)
        sys.modules.pop('jetracer', None)
        if old_package is not None:
            sys.modules['jetracer'] = old_package
        if old_module is not None:
            sys.modules['jetracer.nvidia_racecar'] = old_module
        if old_env is None:
            os.environ.pop('JETRACER_NVIDIA_ROOT', None)
        else:
            os.environ['JETRACER_NVIDIA_ROOT'] = old_env
        shutil.rmtree(temp_root)


def test_nvidia_driver_passes_only_supported_i2c_addresses():
    """Ho tro ca driver chuan (mot dia chi) va image cu (hai dia chi)."""
    class StandardDriver(object):
        i2c_address = 0x40

        @classmethod
        def class_traits(cls):
            return {'i2c_address': object()}

    class LegacyDriver(object):
        i2c_address = 0x40
        i2c_address2 = 0x60

        @classmethod
        def class_traits(cls):
            return {'i2c_address': object(), 'i2c_address2': object()}

    standard = driver_mod._nvidia_constructor_kwargs(
        StandardDriver, i2c_address=0x40, i2c_address2=0x40)
    legacy = driver_mod._nvidia_constructor_kwargs(
        LegacyDriver, i2c_address=0x40, i2c_address2=0x40)

    assert standard == {'i2c_address': 0x40}
    assert legacy == {'i2c_address': 0x40, 'i2c_address2': 0x40}


def test_nvidia_driver_wraps_wrong_0x60_with_actionable_error():
    class BrokenDriver(object):
        i2c_address = 0x40
        i2c_address2 = 0x60

        @classmethod
        def class_traits(cls):
            return {'i2c_address': object(), 'i2c_address2': object()}

        def __init__(self, **_kwargs):
            raise ValueError('No I2C device at address: 0x60')

    original_loader = driver_mod._load_nvidia_racecar
    try:
        driver_mod._load_nvidia_racecar = lambda: (
            BrokenDriver, '/fake/jetracer/nvidia_racecar.py')
        try:
            driver_mod.NvidiaJetRacerDriver(
                i2c_address=0x40, i2c_address2=0x40)
            assert False, 'Phai bao loi khi driver van truy cap 0x60'
        except RuntimeError as exc:
            message = str(exc)
            assert 'i2c_address2=0x40' in message
            assert 'sai bien the driver' in message
            assert 'i2cdetect -y -r 1' in message
    finally:
        driver_mod._load_nvidia_racecar = original_loader


def test_waveshare_direct_driver_stops_and_watchdog_cuts_throttle():
    """Driver 0x40 phai cat ga truc tiep va khong giu lenh cu qua timeout."""
    class FakeServo(object):
        def __init__(self):
            self.history = []
            self.pulse_range = None

        @property
        def throttle(self):
            return self.history[-1] if self.history else None

        @throttle.setter
        def throttle(self, value):
            value = float(value)
            if value < -1.0 or value > 1.0:
                raise ValueError('Out of range')
            self.history.append(value)

        def set_pulse_width_range(self, low, high):
            self.pulse_range = (low, high)

    class FakeChannels(object):
        def __init__(self):
            self.items = [FakeServo() for _ in range(16)]

        def __getitem__(self, index):
            return self.items[index]

    class FakePCA(object):
        def __init__(self):
            self.frequency = None

    class FakeServoKit(object):
        instances = []

        def __init__(self, channels, address):
            self.channels = channels
            self.address = address
            self._pca = FakePCA()
            self.continuous_servo = FakeChannels()
            self.__class__.instances.append(self)

    original_loader = driver_mod._load_servo_kit
    drv = None
    try:
        driver_mod._load_servo_kit = lambda: FakeServoKit
        drv = driver_mod.NvidiaJetRacerDriver(
            implementation='waveshare_single_pca',
            i2c_address=0x40, pwm_frequency=None,
            steering_channel=0, throttle_channel=1,
            steering_gain=-0.65, steering_offset=0.0,
            throttle_gain=0.8, pulse_width_range=(750, 2250),
            command_timeout_s=0.12,
            steering_output_min=-0.40, steering_output_max=0.40)

        kit = FakeServoKit.instances[-1]
        steering = kit.continuous_servo[0]
        throttle = kit.continuous_servo[1]
        assert kit.address == 0x40
        assert kit._pca.frequency is None
        assert steering.pulse_range == (750, 2250)

        drv.set(0.50, 0.25)
        assert abs(steering.throttle - (-0.325)) < 1e-9
        assert abs(throttle.throttle - 0.20) < 1e-9
        drv.set(1.0, 0.0)
        assert abs(steering.throttle - (-0.40)) < 1e-9
        assert abs(drv.last_steering_output - (-0.40)) < 1e-9
        drv.stop()
        assert throttle.throttle == 0.0

        drv.set(0.0, 0.25)
        time.sleep(0.25)
        assert throttle.throttle == 0.0
    finally:
        if drv is not None:
            drv.close()
        driver_mod._load_servo_kit = original_loader


def test_hardware_preflight_initializes_driver_not_only_imports():
    """Preflight phai mo driver that va ghi neutral truoc khi bao OK."""
    from tools import check_hardware

    events = []

    class FakeDriver(object):
        def stop(self):
            events.append('stop')

        def close(self):
            events.append('close')

    original_probe = driver_mod.probe
    original_build = driver_mod.build_driver
    try:
        driver_mod.probe = lambda: {
            'nvidia': 'OK (chi moi truong) - fake',
            'dryrun': 'OK - fake',
        }
        driver_mod.build_driver = lambda kind, cfg: FakeDriver()
        assert check_hardware.check_driver('nvidia', _cfg())
        assert events == ['stop', 'close']
    finally:
        driver_mod.probe = original_probe
        driver_mod.build_driver = original_build


def test_power_mode_parser_and_shutdown_probe_helpers():
    from tools import check_hardware, diagnose_shutdown

    assert check_hardware._is_5w_mode('NV Power Mode: 5W\n1\n')
    assert check_hardware._is_5w_mode('MODE_ID: 1\n')
    assert not check_hardware._is_5w_mode('NV Power Mode: MAXN\n0\n')
    assert isinstance(diagnose_shutdown._uptime_s(), float)
    assert diagnose_shutdown._uptime_s() >= 0.0


def test_latest_grabber_surfaces_background_camera_error():
    """Exception trong thread camera phai hien ra UI, khong duoc chet im lang."""

    class BrokenSource(object):
        def __init__(self):
            self.released = False

        def read(self):
            raise RuntimeError('nvargus test failure')

        def release(self):
            self.released = True

    source = BrokenSource()
    grabber = LatestFrameGrabber(source, wait_timeout_s=0.2).start()
    try:
        frame, frame_id = grabber.read()
        assert frame is None
        assert frame_id == 0
        assert grabber.eof
        assert grabber.error is not None
        assert 'nvargus test failure' in str(grabber.error)
    finally:
        grabber.stop()
    assert source.released


def test_manual_collector_preview_control_and_recording():
    """Integration test UI-camera-control-recorder, khong can ipywidgets that."""

    class FakeWidget(object):
        def __init__(self, value=None, description='', **kwargs):
            self.value = value
            self.description = description
            self.disabled = False
            self._observers = []
            for key, item in kwargs.items():
                setattr(self, key, item)

        def observe(self, callback, names=None):
            # Khong mo phong reactivity that (khong bao gio goi callback khi
            # .value doi) - chi de cac widget dong bo (slider <-> o nhap so)
            # trong ManualDriveCollector khoi tao duoc ma khong AttributeError.
            self._observers.append(callback)

    class FakeLayout(object):
        def __init__(self, **kwargs):
            self.options = kwargs

    class FakeButton(FakeWidget):
        def __init__(self, *args, **kwargs):
            FakeWidget.__init__(self, *args, **kwargs)
            self._callbacks = []

        def on_click(self, callback):
            self._callbacks.append(callback)

    class FakeOutput(FakeWidget):
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc_value, _traceback):
            return False

        def clear_output(self):
            pass

    class FakeAxis(object):
        def __init__(self):
            self.value = 0.0

    class FakeControllerButton(object):
        def __init__(self):
            self.value = 0.0
            self.pressed = False

    class FakeController(FakeWidget):
        def __init__(self, index=0):
            FakeWidget.__init__(self)
            self.index = index
            self.connected = True
            self.name = 'fake-gamepad'
            self.axes = [FakeAxis() for _ in range(4)]
            self.buttons = [FakeControllerButton() for _ in range(10)]

    class FakeBox(FakeWidget):
        def __init__(self, children):
            FakeWidget.__init__(self)
            self.children = children

    fake_widgets = types.ModuleType('ipywidgets.widgets')
    fake_widgets.Controller = FakeController
    fake_widgets.Image = FakeWidget
    fake_widgets.Output = FakeOutput
    fake_widgets.HTML = FakeWidget
    fake_widgets.Text = FakeWidget
    fake_widgets.BoundedIntText = FakeWidget
    fake_widgets.BoundedFloatText = FakeWidget
    fake_widgets.Checkbox = FakeWidget
    fake_widgets.FloatSlider = FakeWidget
    fake_widgets.Button = FakeButton
    fake_widgets.Layout = FakeLayout
    fake_widgets.HBox = FakeBox
    fake_widgets.VBox = FakeBox
    fake_widgets.Label = FakeWidget
    fake_package = types.ModuleType('ipywidgets')
    fake_package.widgets = fake_widgets

    old_package = sys.modules.get('ipywidgets')
    old_widgets = sys.modules.get('ipywidgets.widgets')
    sys.modules['ipywidgets'] = fake_package
    sys.modules['ipywidgets.widgets'] = fake_widgets

    temp_root = tempfile.mkdtemp(prefix='jetracer_ui_collect_')
    collector = None
    try:
        collector = ManualDriveCollector(
            config_path=CONFIG, out_root=temp_root,
            source_kind='synthetic', driver_kind='dryrun')
        collector._ensure_control_thread()
        collector._on_open_camera()

        deadline = time.time() + 2.0
        while time.time() < deadline and not collector.preview.value:
            time.sleep(0.02)
        assert collector._grabber is not None
        assert collector.preview.value

        # ARM lai tren mat dat khong phu thuoc checkbox danh rieng cho actuator.
        collector._on_arm()
        assert collector._armed
        assert not collector.use_deadman.value
        assert not collector.wheels_lifted.value
        assert type(collector._driver).__name__ == 'DryRunDriver'

        collector.controller.buttons[4].pressed = True
        collector.controller.buttons[4].value = 1.0
        collector.controller.axes[2].value = 0.50
        collector.controller.axes[1].value = -1.00
        time.sleep(0.15)
        assert 0.15 < collector._steering_cmd < 0.22
        assert collector._throttle_cmd > 0.15
        assert collector._deadman_pressed

        collector._on_start_recording()
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if collector._writer is not None and collector._writer.count >= 2:
                break
            time.sleep(0.02)
        collector._on_stop_recording()
        assert collector.last_session_dir is not None
        labels = os.path.join(collector.last_session_dir, 'labels.csv')
        video = os.path.join(collector.last_session_dir, 'drive.avi')
        video_sidecar = os.path.join(
            collector.last_session_dir, 'drive.sidecar.csv')
        metadata_path = os.path.join(
            collector.last_session_dir, 'metadata.json')
        assert os.path.exists(labels)
        assert os.path.exists(video) and os.path.getsize(video) > 0
        assert os.path.exists(video_sidecar)
        with open(labels, 'r') as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) >= 2
        assert float(rows[-1]['throttle_cmd']) > 0.15
        with open(video_sidecar, 'r') as fh:
            video_rows = list(csv.DictReader(fh))
        assert len(video_rows) > 0
        assert 'steering_cmd' in video_rows[0]
        with open(metadata_path, 'r') as fh:
            metadata = __import__('json').load(fh)
        assert metadata['video_frames'] == len(video_rows)
        assert metadata['video_error'] is None
        cap = cv2.VideoCapture(video)
        ok, saved_frame = cap.read()
        cap.release()
        assert ok and saved_frame is not None

        # Mo phong control loop dang ghi ga thi emergency den. Lock phai dam
        # bao lenh cuoi cung sau khi set bi treo duoc nha ra van la (0, 0).
        collector.controller.axes[1].value = 0.0
        time.sleep(0.08)

        class BlockingDriver(object):
            def __init__(self):
                self.started = threading.Event()
                self.release = threading.Event()
                self.blocked_once = False
                self.last = (0.0, 0.0)

            def set(self, steering, throttle):
                if throttle > 0.0 and not self.blocked_once:
                    self.blocked_once = True
                    self.started.set()
                    self.release.wait(1.0)
                self.last = (steering, throttle)

            def stop(self):
                self.last = (0.0, 0.0)

            def close(self):
                self.stop()

        blocking = BlockingDriver()
        with collector._driver_io_lock:
            collector._driver = blocking
        collector.controller.axes[1].value = -1.0
        assert blocking.started.wait(1.0)
        emergency = threading.Thread(target=collector._on_emergency)
        emergency.start()
        time.sleep(0.03)
        blocking.release.set()
        emergency.join(timeout=1.0)
        assert not emergency.is_alive()
        assert not collector._armed
        assert blocking.last == (0.0, 0.0)

        # Checkbox ke banh chi thuoc bai test, bo tick khong duoc cat lai that.
        collector._armed = True
        blocking.last = (0.2, 0.2)
        collector._on_wheels_lifted_change({'new': False})
        assert collector._armed
        assert blocking.last == (0.2, 0.2)
    finally:
        if collector is not None:
            collector.close()
        shutil.rmtree(temp_root)
        if old_package is None:
            del sys.modules['ipywidgets']
        else:
            sys.modules['ipywidgets'] = old_package
        if old_widgets is None:
            del sys.modules['ipywidgets.widgets']
        else:
            sys.modules['ipywidgets.widgets'] = old_widgets


def test_lane_detector_tracks_synthetic_road():
    """CTE phai BAM theo tam duong, khong chi nam trong [-1, 1].

    Ban test cu chi assert `-1 <= cte <= 1` va `found >= 30/40` - mot detector
    tra ve hang so 0.0 van pass. Nguon tong hop uon theo sin nen co ground truth
    thuc su; kiem tra tuong quan moi bat duoc loi bam nham vach.
    """
    # Nguon tong hop mo phong sa ban THI: vach trang dut khuc tren lane toi.
    cfg = _cfg(lane__line_color='white')
    det = LaneDetector(cfg)
    src = SyntheticSource(n_frames=120)

    got, truth, found = [], [], 0
    for i in range(120):
        ok, frame = src.read()
        assert ok
        res = det.process(frame)
        assert -1.0 <= res.cte <= 1.0
        if res.found:
            found += 1
        got.append(res.cte)
        # SyntheticSource: center = w/2 + sin(i*0.05 + depth*1.2) * w*0.18
        # tai day anh (depth = 1) -> lech chuan hoa = sin(i*0.05 + 1.2) * 0.36
        truth.append(np.sin(i * 0.05 + 1.2) * 0.36)

    assert found >= 110, 'chi bat duoc vach o %d/120 frame' % found

    got = np.array(got)
    truth = np.array(truth)
    assert got.std() > 0.02, 'cte gan nhu khong doi - detector khong bam gi ca'
    corr = float(np.corrcoef(got, truth)[0, 1])
    assert corr > 0.9, 'cte khong bam tam duong (tuong quan chi %.3f)' % corr


def test_lane_detector_ignores_wrong_colour_line():
    """Doi mau vach ma detector van "tim thay" nghia la no dang bam nham thu khac."""
    cfg = _cfg(lane__line_color='red')      # nguon tong hop chi co vach TRANG
    det = LaneDetector(cfg)
    src = SyntheticSource(n_frames=40)
    found = 0
    for _ in range(40):
        ok, frame = src.read()
        assert ok
        if det.process(frame).found:
            found += 1
    assert found <= 4, 'bat duoc vach do o %d/40 frame nhung anh khong co mau do' % found


def test_lane_detector_survives_dash_gaps():
    """Vach DUT: khong duoc bao mat vach chi vi dang o giua hai net."""
    cfg = _cfg(lane__line_color='white')
    det = LaneDetector(cfg)
    src = SyntheticSource(n_frames=60)
    lost = 0
    for _ in range(60):
        ok, frame = src.read()
        assert ok
        if not det.process(frame).found:
            lost += 1
    assert lost <= 3, 'mat vach %d/60 frame tren duong lien mach' % lost


def test_lane_curvature_and_lookahead_do_not_saturate():
    """Do cong phai nam trong vung dung duoc, khong ghim +-1.

    Bam +-1 nghia la da thuc dang duoc NGOAI SUY ra ngoai vung nhin thay vach;
    khi do ga se tut ve v_min suot luot chay va xe bo het toc do.
    """
    cfg = _cfg(lane__line_color='white')
    det = LaneDetector(cfg)
    src = SyntheticSource(n_frames=120)
    curvs, looks = [], []
    for _ in range(120):
        ok, frame = src.read()
        assert ok
        res = det.process(frame)
        if res.found:
            curvs.append(abs(res.curvature))
            looks.append(abs(res.cte_lookahead))
    assert curvs, 'khong frame nao tim thay vach'
    saturated = float(np.mean([c >= 0.999 for c in curvs]))
    assert saturated < 0.10, 'do cong bao hoa o %.0f%% frame' % (100 * saturated)
    assert float(np.mean([l >= 0.999 for l in looks])) < 0.10


def _synthetic_shaded_frame(w=160, h=120, rg_corner=1.6, bg_corner=1.2):
    """Anh xam trung tinh bi am do dan ve goc - gia lap dung lens shading that."""
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r2 = (((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2) / 2.0
    base = 120.0
    img = np.zeros((h, w, 3), np.float64)
    img[:, :, 1] = base
    img[:, :, 2] = base * (1.0 + (rg_corner - 1.0) * r2)
    img[:, :, 0] = base * (1.0 + (bg_corner - 1.0) * r2)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_shading_corrector_flattens_measured_colour_cast():
    """Hieu chuan tren anh am do -> sau khi sua, R/G phai phang gan 1.0."""
    frame = _synthetic_shaded_frame()
    before = measure_ratio_profile(frame.astype(np.float64), n_bins=6)
    spread_before = max(p['rg'] for p in before) - min(p['rg'] for p in before)
    assert spread_before > 0.3, 'anh test phai co am mau ro ret'

    coeff_r, coeff_b = fit_coefficients(before)
    corrector = ShadingCorrector(coeff_r, coeff_b, enabled=True)
    after = measure_ratio_profile(
        corrector.apply(frame).astype(np.float64), n_bins=6)
    spread_after = max(p['rg'] for p in after) - min(p['rg'] for p in after)

    assert spread_after < 0.2 * spread_before, (
        'bien do R/G chi giam tu %.3f xuong %.3f' % (spread_before, spread_after))


def test_shading_is_off_unless_config_asks():
    """Mac dinh phai TAT. Sua mau am tham lam moi nguong da tune lech vo hinh."""
    from jetracer_baseline.config import Config

    frame = _synthetic_shaded_frame(w=40, h=40)
    off = ShadingCorrector.from_config(Config({}))
    assert off.enabled is False
    assert np.array_equal(off.apply(frame), frame)

    off2 = ShadingCorrector.from_config(
        Config({'camera': {'shading': {'enabled': False}}}))
    assert np.array_equal(off2.apply(frame), frame)

    try:
        ShadingCorrector.from_config(Config(
            {'camera': {'shading': {'enabled': True,
                                    'file': 'khong-ton-tai-abcxyz.yaml'}}}))
    except IOError:
        pass
    else:
        raise AssertionError('phai bao loi khi bat shading ma chua hieu chuan')


def test_shading_apply_resized_matches_apply():
    """Duong nhanh (resize truoc) phai cho cung ket qua voi duong thuong."""
    corrector = ShadingCorrector.from_config(_cfg())
    if not corrector.enabled:
        return
    big = _synthetic_shaded_frame(w=320, h=240)
    assert np.array_equal(corrector.apply(big),
                          corrector.apply_resized(big, (320, 240)))
    out = corrector.apply_resized(_synthetic_shaded_frame(w=640, h=480), (320, 240))
    assert out.shape == (240, 320, 3)


def test_shading_preserves_brightness_and_does_not_clip():
    """Chi doi mau, khong doi do sang -> khong duoc lam chay vung sang."""
    corrector = ShadingCorrector.from_config(_cfg())
    if not corrector.enabled:
        return
    frame = _synthetic_shaded_frame(w=160, h=120)
    out = corrector.apply(frame)
    mean_before, mean_after = float(frame.mean()), float(out.mean())
    assert abs(mean_after - mean_before) / mean_before < 0.15, (
        'do sang trung binh doi tu %.1f sang %.1f' % (mean_before, mean_after))
    assert float(np.mean(out >= 255)) < 0.01, 'qua nhieu pixel bi chay'


def test_tuning_engine_runs_without_ipywidgets():
    """Phan xu ly cua giao dien tune phai chay duoc ngoai Jupyter.

    Neu logic bi tron vao lop widget thi khong test duoc gi - ma logic moi la
    phan de sai. Test nay chinh la ly do LaneTuningEngine tach khoi LaneTuningUI.
    """
    engine = LaneTuningEngine(CONFIG)
    src = SyntheticSource(n_frames=30)
    engine.set_param('lane.line_color', 'white')

    for _ in range(30):
        ok, frame = src.read()
        assert ok
        result = engine.process(frame, 1.0 / 30.0)
        assert set(['proc', 'lane', 'steer', 'throttle']) <= set(result.keys())
        assert -1.0 <= result['steer'] <= 1.0
        assert 0.0 <= result['throttle'] <= engine.v_max + 1e-9

    summary = engine.stats.summary()
    assert summary is not None and summary['n'] == 30

    panel = engine.render_panel(result, fps=20.0, armed=False, width=640)
    assert panel.shape[1] == 640 and panel.ndim == 3
    assert engine.encode_jpeg(panel)


def test_tuning_engine_applies_slider_changes_live():
    """Doi tham so phai co tac dung ngay, khong can tao lai engine."""
    engine = LaneTuningEngine(CONFIG)
    engine.set_param('control.v_max', 0.40)
    engine.set_param('control.curve_slowdown', 0.0)
    engine.set_param('control.slowdown', 0.0)
    src = SyntheticSource(n_frames=5)
    ok, frame = src.read()
    assert ok
    assert abs(engine.process(frame, 0.05)['throttle'] - 0.40) < 1e-6

    engine.set_param('control.v_max', 0.10)
    engine.set_param('control.v_min', 0.05)
    assert abs(engine.process(frame, 0.05)['throttle'] - 0.10) < 1e-6


def test_tuning_engine_saves_only_changed_keys(tmpdir=None):
    """LUU CONFIG phai ghi file override nho, khong de len default.yaml.

    Ghi de default.yaml se mat toan bo comment giai thich vi sao tung con so
    duoc chon - phan dat gia nhat cua file do.
    """
    workdir = tmpdir or tempfile.mkdtemp(prefix='tune_save_')
    try:
        engine = LaneTuningEngine(CONFIG)
        out = os.path.join(workdir, 'tuned.yaml')

        assert engine.save_overrides(out) is None, 'chua doi gi ma da ghi file'
        assert not os.path.exists(out)

        engine.set_param('control.pid.kp', 0.42)
        engine.set_param('lane.hsv_s_min', 111)
        diff = engine.save_overrides(out)

        assert diff == {'control': {'pid': {'kp': 0.42}},
                        'lane': {'hsv_s_min': 111}}, diff
        # File phai nap lai duoc va thuc su de len config goc
        cfg = load_config(CONFIG, [out])
        assert abs(float(cfg.get('control.pid.kp')) - 0.42) < 1e-9
        assert int(cfg.get('lane.hsv_s_min')) == 111
        assert cfg.get('control.v_max') == load_config(CONFIG).get('control.v_max')
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_tuning_saved_thresholds_survive_colour_switch():
    """Nguong S/V da luu KHONG duoc de len preset khi doi mau vach.

    Neu UI ghi thang dai HSV cua mau do, sau nay doi sang `white` thi dai do cu
    van con hieu luc va detector khong bat duoc gi - loi im lang rat kho tim.
    """
    engine = LaneTuningEngine(CONFIG)
    engine.set_param('lane.hsv_s_min', 40)

    engine.set_param('lane.line_color', 'red')
    engine.rebuild()
    red_low = list(engine.lane.hsv_low_1)

    engine.set_param('lane.line_color', 'white')
    engine.rebuild()
    white_low = list(engine.lane.hsv_low_1)

    assert int(red_low[1]) == 40 and int(white_low[1]) == 40, 'S floor phai duoc ap'
    # Hue/V cua hai preset khac han nhau -> phai doi theo mau, khong bi dong bang
    assert int(red_low[2]) != int(white_low[2]), (
        'doi mau vach nhung dai HSV khong doi: %s vs %s' % (red_low, white_low))
    assert engine.lane.hsv_low_2 is None, 'preset trang khong co dai hue thu hai'


def test_tuning_engine_soft_start_ramps_throttle():
    """Bam CHAY khong duoc cho ga nhay thang len muc chay.

    Ga nhay tu 0 len v_max lam banh truot va nguoi bam khong kip phan ung neu
    chieu lai dang sai - dung luc nguy hiem nhat.
    """
    engine = LaneTuningEngine(CONFIG)
    engine.set_param('lane.line_color', 'white')
    engine.soft_start_s = 1.0
    src = SyntheticSource(n_frames=5)
    ok, frame = src.read()
    assert ok

    # Chua chay -> preview hien ga day du, khong ramp
    full = engine.process(frame, 0.05)['throttle']
    assert full > 0.0

    t0 = 1000.0
    engine.start_run(now=t0)
    assert engine.running
    values = [engine.process(frame, 0.05, now=t0 + e)['throttle']
              for e in (0.0, 0.25, 0.5, 1.0, 2.0)]
    assert values[0] == 0.0, 'ga phai bat dau tu 0'
    for i in range(1, 4):
        assert values[i] > values[i - 1], 'ga phai tang dan: %r' % (values,)
    assert abs(values[3] - full) < 1e-6, 'sau soft_start_s phai dat ga day du'
    assert abs(values[4] - full) < 1e-6

    engine.stop_run()
    assert not engine.running
    assert abs(engine.process(frame, 0.05)['throttle'] - full) < 1e-6


def test_tuning_engine_soft_start_disabled_is_immediate():
    engine = LaneTuningEngine(CONFIG)
    engine.set_param('lane.line_color', 'white')
    engine.soft_start_s = 0.0
    src = SyntheticSource(n_frames=3)
    ok, frame = src.read()
    assert ok
    full = engine.process(frame, 0.05)['throttle']
    engine.start_run(now=500.0)
    assert abs(engine.process(frame, 0.05, now=500.0)['throttle'] - full) < 1e-6


def test_rolling_stats_window_and_flip_count():
    stats = RollingStats(window=5)
    for value in (0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1):
        stats.push(value, value, True, 4)
    assert stats.n == 5, 'cua so truot phai gioi han so mau'
    summary = stats.summary()
    assert summary['steer_flips'] == 4
    assert summary['loss_pct'] == 0.0

    stats.reset()
    assert stats.summary() is None


def test_dict_diff_ignores_unchanged_nested_keys():
    base = {'a': {'b': 1, 'c': 2}, 'd': 3}
    assert _dict_diff(base, {'a': {'b': 1, 'c': 2}, 'd': 3}) == {}
    assert _dict_diff(base, {'a': {'b': 9, 'c': 2}, 'd': 3}) == {'a': {'b': 9}}
    assert _dict_diff(base, {'a': {'b': 1, 'c': 2}, 'd': 3, 'e': 5}) == {'e': 5}


def test_sign_tracker_needs_votes():
    """Mot frame nhieu KHONG duoc lam thay doi quyet dinh o nga tu."""
    cfg = _cfg(signs__vote_k=3, signs__vote_n=5)
    tracker = SignTracker(cfg)

    class D(object):
        def __init__(self, label, conf, area):
            self.label, self.confidence, self.area_ratio = label, conf, area

    assert tracker.update([D('turn_left', 0.9, 0.05)]) is None   # 1/3
    assert tracker.update([D('turn_left', 0.9, 0.05)]) is None   # 2/3
    out = tracker.update([D('turn_left', 0.9, 0.05)])            # 3/3 -> latch
    assert out is not None and out[0] == 'turn_left'


def test_green_light_needs_higher_confidence_than_red():
    """Bat doi xung an toan: bao nham xanh = vuot den do = HUY LUOT."""
    cfg = _cfg()
    tracker = SignTracker(cfg)

    class D(object):
        def __init__(self, label, conf, area):
            self.label, self.confidence, self.area_ratio = label, conf, area

    conf = 0.5   # nam giua conf_red (0.35) va conf_green (0.80)
    for _ in range(5):
        red = tracker.update([D('red_light', conf, 0.001)])
    assert red is not None and red[0] == 'red_light'   # do: chap nhan

    tracker2 = SignTracker(cfg)
    for _ in range(5):
        green = tracker2.update([D('green_light', conf, 0.001)])
    assert green is None                               # xanh: tu choi


def test_fsm_stops_on_red_and_does_not_creep():
    cfg = _cfg()
    fsm = fsm_mod.DecisionFSM(cfg, task='smartcity')
    fsm.start()

    class Lane(object):
        found, cte, curvature, n_pixels = True, 0.0, 0.0, 500

    class Stop(object):
        found, distance = True, 0.10   # da toi sat vach

    cmd = fsm.step(Lane(), stopline=Stop(), sign=('red_light', 0.9, time.time()))
    assert cmd.force_stop is True
    assert cmd.decision == 'stop'
    assert fsm.state == fsm_mod.S_WAIT_RED

    # Khong thay den nua -> VAN phai dung (nguyen tac: khong chac chan thi dung)
    cmd = fsm.step(Lane(), stopline=Stop(), sign=None)
    assert cmd.force_stop is True


def test_fsm_goes_on_green():
    cfg = _cfg(fsm__stop_hold_s=0.0)
    fsm = fsm_mod.DecisionFSM(cfg, task='smartcity')
    fsm.start()

    class Lane(object):
        found, cte, curvature, n_pixels = True, 0.0, 0.0, 500

    class Stop(object):
        found, distance = True, 0.10

    fsm.step(Lane(), stopline=Stop(), sign=('red_light', 0.9, time.time()))
    cmd = fsm.step(Lane(), stopline=Stop(), sign=('green_light', 0.9, time.time()))
    assert cmd.force_stop is False
    assert cmd.throttle_scale > 0


def test_fsm_mandatory_sign_turns():
    cfg = _cfg()
    fsm = fsm_mod.DecisionFSM(cfg, task='smartcity')
    fsm.start()

    class Lane(object):
        found, cte, curvature, n_pixels = True, 0.0, 0.0, 500

    cmd = fsm.step(Lane(), stopline=None, sign=('turn_left', 0.9, time.time()))
    assert cmd.decision == 'left'
    assert cmd.steer_override is not None and cmd.steer_override < 0


def test_full_pipeline_writes_valid_log(tmpdir=None):
    from jetracer_baseline.logging_csv import FIELDS

    cfg = _cfg()
    log_dir = os.path.join(ROOT, 'logs', 'test')
    runner = Runner(cfg, task='speed', driver_kind='dryrun',
                    source=SyntheticSource(n_frames=60), log_dir=log_dir,
                    verbose=False, sync=True, realtime=False)
    summary = runner.run(max_frames=60)

    assert summary['frames'] == 60
    with open(summary['log']) as fh:
        lines = fh.read().strip().split('\n')
    assert lines[0].split(',') == FIELDS
    assert len(lines) == 62                       # header + 60 frame + 1 dong ket
    # Moi dong du so cot -> parse duoc bang pandas/csv khi viet paper
    for line in lines[1:]:
        assert len(line.split(',')) == len(FIELDS)


if __name__ == '__main__':
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print('PASS  ' + name)
            except Exception as exc:
                failures += 1
                print('FAIL  %s : %s' % (name, exc))
    print('')
    print('%d loi' % failures)
    sys.exit(1 if failures else 0)
