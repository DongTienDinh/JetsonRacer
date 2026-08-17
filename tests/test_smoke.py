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
import time
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from jetracer_baseline import fsm as fsm_mod                     # noqa: E402
from jetracer_baseline.camera import (                          # noqa: E402
    CSISource, LatestFrameGrabber, SyntheticSource,
    gstreamer_pipeline)
from jetracer_baseline.config import load_config                 # noqa: E402
from jetracer_baseline.control.pid import PID                    # noqa: E402
from jetracer_baseline.perception.lane import LaneDetector       # noqa: E402
from jetracer_baseline.perception.signs import SignTracker       # noqa: E402
from jetracer_baseline.pipeline import Runner                    # noqa: E402
from jetracer_baseline.manual_collection import (                # noqa: E402
    DatasetSessionWriter, ManualDriveCollector, apply_deadzone)

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
            startup_attempts=2, startup_delay_s=0.0)
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


def test_jetson_gstreamer_pipeline_has_low_latency_caps():
    pipeline = gstreamer_pipeline(640, 480, 30, flip_method=2)
    assert 'nvarguscamerasrc' in pipeline
    assert 'memory:NVMM' in pipeline
    assert 'format=(string)NV12' in pipeline
    assert 'nvvidconv flip-method=2' in pipeline
    assert 'format=(string)BGR' in pipeline
    assert 'drop=true' in pipeline
    assert 'max-buffers=1' in pipeline
    assert 'sync=false' in pipeline


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
            for key, item in kwargs.items():
                setattr(self, key, item)

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

        collector._on_arm()
        assert collector._armed
        assert type(collector._driver).__name__ == 'DryRunDriver'

        collector.controller.buttons[4].pressed = True
        collector.controller.buttons[4].value = 1.0
        collector.controller.axes[2].value = 0.50
        collector.controller.axes[1].value = -1.00
        time.sleep(0.15)
        assert collector._steering_cmd > 0.40
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
        assert os.path.exists(labels)
        with open(labels, 'r') as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) >= 2
        assert float(rows[-1]['throttle_cmd']) > 0.15
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
    cfg = _cfg()
    det = LaneDetector(cfg)
    src = SyntheticSource(n_frames=40)
    found = 0
    for _ in range(40):
        ok, frame = src.read()
        assert ok
        res = det.process(frame)
        assert -1.0 <= res.cte <= 1.0
        if res.found:
            found += 1
    # Duong tong hop ro rang -> phai bat duoc phan lon frame
    assert found >= 30


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
