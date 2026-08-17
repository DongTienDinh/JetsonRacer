# -*- coding: utf-8 -*-
"""Adapter dieu khien chap hanh (servo lai + motor).

Backend DA CHOT la `nvidia` - theo `control.txt` cua BTC (dung thu vien
NVIDIA jetracer, `NvidiaRacecar`, `steering_gain = -1.0`). `ros` van duoc
giu lai o day phong khi image doi khac, nhung KHONG phai duong mac dinh nua:

  1. `nvidia_jetracer` : thu vien NVIDIA jetracer (DA CHOT, dung backend nay)
        from jetracer.nvidia_racecar import NvidiaRacecar
        car.steering = x ; car.throttle = y
  2. `ros`             : ban ROS cua Waveshare, publish geometry_msgs/Twist len /cmd_vel - du phong, chua dung
  3. `dryrun`          : khong dieu khien gi, chi ghi log (dev tren laptop)

Ngay dau tien co xe: van nen chay `python3 -m src.jetracer_baseline.cli probe`
de xac nhan thu vien `jetracer` co san truoc khi chay `--driver nvidia`.

Quy uoc chung:
    steering  in [-1, 1]   (-1 = het trai, +1 = het phai)
    throttle  in [-1, 1]   (am = lui)
"""

import importlib
import os
import sys


def _nvidia_repo_candidates():
    """Vi tri thuong gap cua NVIDIA-AI-IOT/jetracer tren image Jetson."""
    paths = []
    configured = os.environ.get('JETRACER_NVIDIA_ROOT', '')
    if configured:
        paths.extend([p for p in configured.split(os.pathsep) if p])
    home = os.path.expanduser('~')
    paths.extend([
        os.path.join(home, 'jetracer'),
        '/home/jetson/jetracer',
        os.path.join(home, 'jetcard', 'jetracer'),
        '/opt/jetracer',
    ])
    result = []
    for path in paths:
        path = os.path.abspath(path)
        module_path = os.path.join(path, 'jetracer', 'nvidia_racecar.py')
        if path not in result and os.path.isfile(module_path):
            result.append(path)
    return result


def _load_nvidia_racecar():
    """Import NvidiaRacecar, tu them sibling repo vao sys.path neu chua install."""
    first_error = None
    try:
        module = importlib.import_module('jetracer.nvidia_racecar')
        return module.NvidiaRacecar, getattr(module, '__file__', '?')
    except Exception as exc:
        first_error = exc

    tried = []
    for path in _nvidia_repo_candidates():
        tried.append(path)
        if path not in sys.path:
            sys.path.insert(0, path)
        # Xoa package import do dang neu Python da tim thay mot package trung ten.
        sys.modules.pop('jetracer.nvidia_racecar', None)
        sys.modules.pop('jetracer', None)
        importlib.invalidate_caches()
        try:
            module = importlib.import_module('jetracer.nvidia_racecar')
            return module.NvidiaRacecar, getattr(module, '__file__', '?')
        except Exception as exc:
            first_error = exc

    raise ImportError(
        'Khong import duoc jetracer.nvidia_racecar: %s. Da tim: %s. '
        'Neu repo nam noi khac, dat JETRACER_NVIDIA_ROOT=/duong/dan/toi/repo.' %
        (first_error, ', '.join(tried) if tried else '(khong tim thay repo)'))


def _clip(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


class BaseDriver(object):
    def set(self, steering, throttle):
        raise NotImplementedError

    def stop(self):
        self.set(0.0, 0.0)

    def close(self):
        self.stop()


class DryRunDriver(BaseDriver):
    """Khong gui lenh xuong phan cung. Giu lai lenh cuoi de log/test."""

    def __init__(self):
        self.last = (0.0, 0.0)

    def set(self, steering, throttle):
        self.last = (_clip(steering, -1.0, 1.0), _clip(throttle, -1.0, 1.0))


class NvidiaJetRacerDriver(BaseDriver):
    """`None` o bat ky tham so nao nghia la "khong dung toi thuoc tinh do" -
    giu nguyen mac dinh cua thu vien jetracer, KHONG phai ep ve 0.

    steering_gain/offset/throttle_gain: xem BASELINE.md / control.txt cua BTC
    de biet gia tri chot cho ngay thi. pulse_width_range: (low, high) truyen
    thang cho `steering_motor.set_pulse_width_range`, hoac None de khong goi.
    """

    def __init__(self, steering_gain=None, steering_offset=None,
                 throttle_gain=None, pulse_width_range=None):
        NvidiaRacecar, module_path = _load_nvidia_racecar()

        self.car = NvidiaRacecar()
        print('[driver] NvidiaRacecar: %s' % module_path)
        print('[driver] steering_gain mac dinh cua thu vien: %r' % (
            self.car.steering_gain,))

        if pulse_width_range is not None:
            try:
                self.car.steering_motor.set_pulse_width_range(*pulse_width_range)
                print('[driver] set_pulse_width_range%r : OK' % (
                    tuple(pulse_width_range),))
            except Exception as exc:
                print('[driver] set_pulse_width_range%r : LOI - %s' % (
                    tuple(pulse_width_range), exc))

        if steering_gain is not None:
            self.car.steering_gain = steering_gain
        if steering_offset is not None:
            self.car.steering_offset = steering_offset
        if throttle_gain is not None:
            self.car.throttle_gain = throttle_gain

    def set(self, steering, throttle):
        self.car.steering = _clip(steering, -1.0, 1.0)
        self.car.throttle = _clip(throttle, -1.0, 1.0)


class RosCmdVelDriver(BaseDriver):
    """Publish /cmd_vel. `linear.x` = toc do tien, `angular.z` = toc do quay.

    `max_linear` / `max_angular` phai do lai tren xe that - gia tri duoi day chi
    la cho trong, KHONG duoc dung khi chua hieu chuan.
    """

    def __init__(self, topic='/cmd_vel', max_linear=0.6, max_angular=2.0):
        import rospy  # noqa
        from geometry_msgs.msg import Twist  # noqa

        self._Twist = Twist
        if not rospy.core.is_initialized():
            rospy.init_node('jetracer_baseline', anonymous=True, disable_signals=True)
        self.pub = rospy.Publisher(topic, Twist, queue_size=1)
        self.max_linear = max_linear
        self.max_angular = max_angular

    def set(self, steering, throttle):
        msg = self._Twist()
        msg.linear.x = _clip(throttle, -1.0, 1.0) * self.max_linear
        # steering duong = re phai = angular.z am (quy uoc tay phai cua ROS)
        msg.angular.z = -_clip(steering, -1.0, 1.0) * self.max_angular
        self.pub.publish(msg)


def build_driver(kind, cfg=None):
    if kind == 'nvidia':
        if cfg is None:
            return NvidiaJetRacerDriver()
        return NvidiaJetRacerDriver(
            steering_gain=cfg.get('control.driver.steering_gain'),
            steering_offset=cfg.get('control.driver.steering_offset'),
            throttle_gain=cfg.get('control.driver.throttle_gain'),
            pulse_width_range=cfg.get('control.driver.pulse_width_range'),
        )
    if kind == 'ros':
        return RosCmdVelDriver()
    return DryRunDriver()


def probe():
    """Bao cao backend nao kha dung tren may hien tai."""
    report = {}
    try:
        _cls, module_path = _load_nvidia_racecar()
        report['nvidia'] = 'OK - thu vien jetracer: %s' % module_path
    except Exception as exc:
        report['nvidia'] = 'KHONG - ' + str(exc)
    try:
        import rospy  # noqa
        import geometry_msgs.msg  # noqa
        report['ros'] = 'OK - rospy + geometry_msgs co san'
    except Exception as exc:
        report['ros'] = 'KHONG - ' + str(exc)
    report['dryrun'] = 'OK - luon co'
    return report
