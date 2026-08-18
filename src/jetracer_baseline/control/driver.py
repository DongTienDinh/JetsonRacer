# -*- coding: utf-8 -*-
"""Adapter dieu khien chap hanh (servo lai + motor).

Backend DA CHOT la `nvidia`. Voi config mac dinh, ten backend nay dung ServoKit
truc tiep tren PCA9685 0x40 cua Waveshare (channel 0 lai, channel 1 ga), khong
di qua bien the NvidiaRacecar hai dia chi cua image. `ros` va adapter thu vien
cu van duoc giu lai lam du phong:

  1. `nvidia`          : ServoKit/PCA9685 0x40 truc tiep (mac dinh)
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
import threading
import time


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


def _load_servo_kit():
    """Nap ServoKit rieng de co the test driver tren laptop bang fake class."""
    from adafruit_servokit import ServoKit
    return ServoKit


def _clip(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def _nvidia_constructor_kwargs(cls, i2c_address=None, i2c_address2=None):
    """Tao kwargs I2C phu hop voi dung bien the ``NvidiaRacecar`` dang cai.

    NVIDIA/Waveshare da phat hanh nhieu bien the cung ten class. Ban chuan va
    ban ``ws/pro`` dung ``i2c_address``; mot so image cu co them
    ``i2c_address2`` va mac dinh no la 0x60. Chi truyen trait ma class thuc su
    ho tro de khong lam hong kha nang tuong thich voi cac image con lai.
    """
    supported = set()
    try:
        supported.update(cls.class_traits().keys())
    except (AttributeError, TypeError):
        pass

    for name in ('i2c_address', 'i2c_address2'):
        if hasattr(cls, name):
            supported.add(name)

    requested = {
        'i2c_address': i2c_address,
        'i2c_address2': i2c_address2,
    }
    return dict((name, value) for name, value in requested.items()
                if value is not None and name in supported)


def _format_i2c_address(value):
    try:
        return '0x%02x' % int(value)
    except (TypeError, ValueError):
        return repr(value)


def _nvidia_init_error(exc, module_path, constructor_kwargs):
    """Doi loi I2C kho hieu thanh huong dan co the kiem tra ngay tren xe."""
    requested = ', '.join(
        '%s=%s' % (name, _format_i2c_address(value))
        for name, value in sorted(constructor_kwargs.items()))
    if not requested:
        requested = '(driver khong cong khai tham so dia chi I2C)'

    detail = str(exc)
    extra = ''
    requested_0x40 = any(
        '0x40' in _format_i2c_address(value).lower()
        for value in constructor_kwargs.values())
    if '0x60' in detail.lower() and requested_0x40:
        extra = (
            ' Driver dang import van co truy cap mac dinh 0x60; day thuong la '
            'sai bien the driver cho JetRacer Pro/Waveshare.')

    return RuntimeError(
        'Khong khoi tao duoc NvidiaRacecar. module=%s; dia chi cau hinh: %s; '
        'loi goc: %s.%s Chay `sudo i2cdetect -y -r 1`: mach JetRacer Pro '
        'phai xuat hien dia chi 40 trong bang. Neu co 0x40 ma van loi, cai dung nhanh '
        'Waveshare `ws/pro` va khoi dong lai Jupyter kernel.' %
        (module_path, requested, detail, extra))


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
    """Driver cho JetRacer, co hai implementation.

    ``waveshare_single_pca`` dieu khien truc tiep PCA9685 duy nhat o 0x40 bang
    ServoKit: channel 0 lai, channel 1 ga. Day la duong mac dinh cho xe cua BTC.
    No khong dung bien the NvidiaRacecar hai dia chi (0x40 + 0x60), vi ep ca
    hai object do vao 0x40 lam hai driver tranh chap tan so/channel va co the
    khien motor chay lien tuc sau loi ``Out of range``.

    ``library`` giu adapter NvidiaRacecar cu lam du phong cho image khac.

    steering_gain/offset/throttle_gain: xem BASELINE.md / control.txt cua BTC
    de biet gia tri chot cho ngay thi. pulse_width_range: (low, high) truyen
    thang cho `steering_motor.set_pulse_width_range`, hoac None de khong goi.

    Moi lenh phai duoc lam moi truoc ``command_timeout_s``. Neu thread UI bi
    dung, watchdog ghi ga=0 truc tiep, khong cho xe giu lenh cu vo thoi han.
    """

    def __init__(self, steering_gain=None, steering_offset=None,
                 throttle_gain=None, pulse_width_range=None,
                 i2c_address=None, i2c_address2=None,
                 implementation='library', pwm_frequency=None,
                 steering_channel=0, throttle_channel=1,
                 command_timeout_s=0.8, steering_output_min=None,
                 steering_output_max=None):
        self.implementation = implementation or 'library'
        self._io_lock = threading.RLock()
        self._watchdog_stop = threading.Event()
        self._watchdog_thread = None
        self._last_command_time = None
        self._closed = False
        self.command_timeout_s = max(0.0, float(command_timeout_s or 0.0))
        self.steering_output_min = -1.0 if steering_output_min is None \
            else float(steering_output_min)
        self.steering_output_max = 1.0 if steering_output_max is None \
            else float(steering_output_max)
        if self.steering_output_min < -1.0 or \
                self.steering_output_max > 1.0 or \
                self.steering_output_min > self.steering_output_max:
            raise ValueError(
                'Gioi han output servo khong hop le: min=%s max=%s' %
                (self.steering_output_min, self.steering_output_max))
        self.last_steering_output = 0.0
        self.last_throttle_output = 0.0

        if self.implementation == 'waveshare_single_pca':
            self._init_waveshare_single_pca(
                steering_gain=steering_gain,
                steering_offset=steering_offset,
                throttle_gain=throttle_gain,
                pulse_width_range=pulse_width_range,
                i2c_address=i2c_address,
                pwm_frequency=pwm_frequency,
                steering_channel=steering_channel,
                throttle_channel=throttle_channel)
        elif self.implementation == 'library':
            self._init_library(
                steering_gain=steering_gain,
                steering_offset=steering_offset,
                throttle_gain=throttle_gain,
                pulse_width_range=pulse_width_range,
                i2c_address=i2c_address,
                i2c_address2=i2c_address2)
        else:
            raise ValueError('implementation driver khong hop le: %s' %
                             self.implementation)

        if self.command_timeout_s > 0.0:
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop, name='jetracer-driver-watchdog')
            self._watchdog_thread.daemon = True
            self._watchdog_thread.start()

    def _init_library(self, steering_gain, steering_offset, throttle_gain,
                      pulse_width_range, i2c_address, i2c_address2):
        NvidiaRacecar, module_path = _load_nvidia_racecar()

        print('[driver] NvidiaRacecar: %s' % module_path)
        constructor_kwargs = _nvidia_constructor_kwargs(
            NvidiaRacecar, i2c_address=i2c_address,
            i2c_address2=i2c_address2)
        if constructor_kwargs:
            print('[driver] I2C: %s' % ', '.join(
                '%s=%s' % (name, _format_i2c_address(value))
                for name, value in sorted(constructor_kwargs.items())))
        try:
            self.car = NvidiaRacecar(**constructor_kwargs)
        except Exception as exc:
            raise _nvidia_init_error(exc, module_path, constructor_kwargs)
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
        self.steering_gain = float(self.car.steering_gain)
        self.steering_offset = float(self.car.steering_offset)
        self.throttle_gain = float(self.car.throttle_gain)
        print('[driver] hard steering output limit: %.3f .. %.3f' % (
            self.steering_output_min, self.steering_output_max))

    def _init_waveshare_single_pca(self, steering_gain, steering_offset,
                                   throttle_gain, pulse_width_range,
                                   i2c_address, pwm_frequency,
                                   steering_channel, throttle_channel):
        ServoKit = _load_servo_kit()
        address = 0x40 if i2c_address is None else int(i2c_address)
        self.steering_gain = -0.65 if steering_gain is None \
            else float(steering_gain)
        self.steering_offset = 0.0 if steering_offset is None \
            else float(steering_offset)
        self.throttle_gain = 0.8 if throttle_gain is None \
            else float(throttle_gain)
        self.steering_channel = int(steering_channel)
        self.throttle_channel = int(throttle_channel)

        print('[driver] Waveshare single PCA9685: address=%s, steering=ch%d, '
              'throttle=ch%d' % (
                  _format_i2c_address(address), self.steering_channel,
                  self.throttle_channel))
        self.kit = ServoKit(channels=16, address=address)
        if pwm_frequency is not None:
            self.kit._pca.frequency = int(pwm_frequency)
            print('[driver] PCA9685 frequency=%d Hz' % int(pwm_frequency))
        self.steering_motor = self.kit.continuous_servo[
            self.steering_channel]
        self.throttle_motor = self.kit.continuous_servo[
            self.throttle_channel]

        if pulse_width_range is not None:
            self.steering_motor.set_pulse_width_range(*pulse_width_range)
            print('[driver] set_pulse_width_range%r : OK' %
                  (tuple(pulse_width_range),))
        print('[driver] hard steering output limit: %.3f .. %.3f' % (
            self.steering_output_min, self.steering_output_max))

        # Ghi neutral ngay sau khoi tao, truoc khi bat dau watchdog/UI.
        with self._io_lock:
            self._force_stop_direct_locked()

    def _force_stop_direct_locked(self):
        """Cat ga truoc, sau do moi dua lai ve giua."""
        throttle_error = None
        try:
            self.throttle_motor.throttle = 0.0
        except Exception as exc:
            throttle_error = exc
        try:
            self.steering_motor.throttle = _clip(
                self.steering_offset, self.steering_output_min,
                self.steering_output_max)
        except Exception:
            if throttle_error is None:
                raise
        else:
            self.last_steering_output = _clip(
                self.steering_offset, self.steering_output_min,
                self.steering_output_max)
            self.last_throttle_output = 0.0
        if throttle_error is not None:
            raise throttle_error

    def _set_direct_locked(self, steering, throttle):
        steering_output = _clip(
            float(steering) * self.steering_gain + self.steering_offset,
            self.steering_output_min, self.steering_output_max)
        throttle_output = _clip(
            float(throttle) * self.throttle_gain, -1.0, 1.0)
        try:
            # Chi cap ga sau khi lenh lai hop le da duoc ghi.
            self.steering_motor.throttle = steering_output
            self.throttle_motor.throttle = throttle_output
            self.last_steering_output = steering_output
            self.last_throttle_output = throttle_output
        except Exception:
            # Bat ky loi nao cung phai thu cat ga truoc khi day exception len UI.
            try:
                self._force_stop_direct_locked()
            except Exception:
                pass
            raise

    def _stop_locked(self):
        if self.implementation == 'waveshare_single_pca':
            self._force_stop_direct_locked()
        else:
            # Ga ve 0 truoc. Khong dung BaseDriver.stop() de tranh mot observer
            # steering loi lam bo qua lenh cat ga.
            self.car.throttle = 0.0
            self.car.steering = 0.0
            self.last_throttle_output = 0.0
            self.last_steering_output = _clip(
                self.steering_offset, self.steering_output_min,
                self.steering_output_max)

    def _limit_library_steering(self, steering):
        """Gioi han output servo cua NvidiaRacecar ma van giu API steering."""
        if abs(self.steering_gain) < 1e-12:
            return float(steering)
        output = _clip(
            float(steering) * self.steering_gain + self.steering_offset,
            self.steering_output_min, self.steering_output_max)
        return _clip(
            (output - self.steering_offset) / self.steering_gain,
            -1.0, 1.0)

    def _watchdog_loop(self):
        interval = min(0.10, max(0.02, self.command_timeout_s / 4.0))
        while not self._watchdog_stop.wait(interval):
            expired = False
            stop_error = None
            with self._io_lock:
                if self._last_command_time is not None and \
                        (time.monotonic() - self._last_command_time) > \
                        self.command_timeout_s:
                    expired = True
                    try:
                        self._stop_locked()
                    except Exception as exc:
                        # Giu timestamp qua han de vong sau thu cat ga lai.
                        stop_error = exc
                    else:
                        self._last_command_time = None
            if expired:
                if stop_error is None:
                    print('[driver] WATCHDOG: mat lenh %.2f s -> throttle=0' %
                          self.command_timeout_s)
                else:
                    print('[driver] WATCHDOG: LOI cat ga, se thu lai: %s' %
                          stop_error)

    def set(self, steering, throttle):
        with self._io_lock:
            if self._closed:
                raise RuntimeError('Driver da dong')
            steering = _clip(float(steering), -1.0, 1.0)
            throttle = _clip(float(throttle), -1.0, 1.0)
            if self.implementation == 'waveshare_single_pca':
                self._set_direct_locked(steering, throttle)
            else:
                steering = self._limit_library_steering(steering)
                self.car.steering = steering
                self.car.throttle = throttle
                self.last_steering_output = _clip(
                    steering * self.steering_gain + self.steering_offset,
                    self.steering_output_min, self.steering_output_max)
                self.last_throttle_output = _clip(
                    throttle * self.throttle_gain, -1.0, 1.0)
            self._last_command_time = time.monotonic()

    def stop(self):
        with self._io_lock:
            try:
                self._stop_locked()
            except Exception:
                # Buoc watchdog vao trang thai qua han de no tiep tuc thu cat
                # ga neu loi I2C chi la tam thoi.
                if self.command_timeout_s > 0.0:
                    self._last_command_time = (
                        time.monotonic() - self.command_timeout_s - 1.0)
                raise
            else:
                self._last_command_time = None

    def close(self):
        try:
            self.stop()
        finally:
            with self._io_lock:
                self._closed = True
            self._watchdog_stop.set()
            if self._watchdog_thread is not None and \
                    self._watchdog_thread is not threading.current_thread():
                self._watchdog_thread.join(timeout=1.0)


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
            implementation=cfg.get(
                'control.driver.implementation', 'library'),
            steering_gain=cfg.get('control.driver.steering_gain'),
            steering_offset=cfg.get('control.driver.steering_offset'),
            throttle_gain=cfg.get('control.driver.throttle_gain'),
            pulse_width_range=cfg.get('control.driver.pulse_width_range'),
            i2c_address=cfg.get('control.driver.i2c_address'),
            i2c_address2=cfg.get('control.driver.i2c_address2'),
            pwm_frequency=cfg.get('control.driver.pwm_frequency'),
            steering_channel=cfg.get('control.driver.steering_channel', 0),
            throttle_channel=cfg.get('control.driver.throttle_channel', 1),
            command_timeout_s=cfg.get(
                'control.driver.command_timeout_s', 0.8),
            steering_output_min=cfg.get(
                'control.driver.steering_output_min'),
            steering_output_max=cfg.get(
                'control.driver.steering_output_max'),
        )
    if kind == 'ros':
        return RosCmdVelDriver()
    return DryRunDriver()


def probe():
    """Bao cao backend nao kha dung tren may hien tai."""
    report = {}
    try:
        ServoKit = _load_servo_kit()
        report['nvidia'] = 'OK (chi moi truong) - ServoKit co san: %s' % (
            getattr(sys.modules.get(ServoKit.__module__), '__file__',
                    ServoKit.__module__),)
    except Exception as exc:
        try:
            _cls, module_path = _load_nvidia_racecar()
            report['nvidia'] = 'OK (chi moi truong) - jetracer co san: %s' % module_path
        except Exception as legacy_exc:
            report['nvidia'] = 'KHONG - ServoKit=%s; jetracer=%s' % (
                exc, legacy_exc)
    try:
        import rospy  # noqa
        import geometry_msgs.msg  # noqa
        report['ros'] = 'OK - rospy + geometry_msgs co san'
    except Exception as exc:
        report['ros'] = 'KHONG - ' + str(exc)
    report['dryrun'] = 'OK - luon co'
    return report
