# -*- coding: utf-8 -*-
"""Bring-up xe theo tung buoc, tu an toan den rui ro.

    python3 tools/check_hardware.py                  # buoc 1-5, KHONG chay motor
    python3 tools/check_hardware.py --driver nvidia --actuator --wheels-are-lifted

    # Tim gioi han co khi that de cua gat hon (xem calibrate_steering()):
    python3 tools/check_hardware.py --driver nvidia --calibrate-steering \
        --wheels-are-lifted

Thu tu cac buoc co chu dich: moi buoc chi chay khi buoc truoc da qua. Buoc 6
(dong co) va buoc 7 (hieu chuan lai) bi khoa sau co --wheels-are-lifted vi xe
co the lao di / banh cham gam.

!! TRUOC KHI CHAY BUOC 6 HOAC 7: KE XE LEN GIA, BANH KHONG CHAM DAT !!
"""

from __future__ import print_function

import argparse
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

PASS = '[ OK ]'
FAIL = '[FAIL]'
WARN = '[WARN]'


def step(n, title):
    print('')
    print('-' * 66)
    print('BUOC %d: %s' % (n, title))
    print('-' * 66)


def check_python():
    step(1, 'Moi truong Python')
    ver = sys.version_info
    print('  Python : %d.%d.%d' % (ver[0], ver[1], ver[2]))
    print('  Platform: %s %s' % (platform.system(), platform.machine()))

    if ver[0] < 3:
        print('%s Dang chay Python 2. Dung python3.' % FAIL)
        return False
    if ver[:2] == (3, 6):
        print('%s Python 3.6 - dung ban tren Jetson Nano JetPack 4.6.' % PASS)
    else:
        print('%s Khong phai 3.6. Neu day la Jetson thi dang dung sai '
              'interpreter; chay tools/check_py36.py truoc khi dong bo code.' % WARN)

    ok = True
    for name in ('numpy', 'cv2', 'yaml'):
        try:
            mod = __import__(name)
            ver_s = getattr(mod, '__version__', '?')
            print('%s %-6s %s' % (PASS, name, ver_s))
        except ImportError as exc:
            print('%s %-6s %s' % (FAIL, name, exc))
            ok = False

    try:
        import cv2
        info = cv2.getBuildInformation()
        for key in ('GStreamer', 'CUDA'):
            line = [l for l in info.split('\n') if l.strip().startswith(key)]
            if line:
                print('       cv2 %s' % line[0].strip())
    except Exception:
        pass
    return ok


def _is_5w_mode(output):
    """Nhan dang output cua cac phien ban nvpmodel tren JetPack 4.x."""
    text = str(output).lower()
    if '5w' in text:
        return True
    if 'mode_id: 1' in text or 'mode id: 1' in text:
        return True
    return any(line.strip() == '1' for line in text.splitlines())


def check_power_mode():
    step(2, 'Nguon Jetson / nvpmodel')
    if platform.system().lower() != 'linux':
        print('%s Khong phai Linux/Jetson - bo qua power mode.' % WARN)
        return None
    try:
        output = subprocess.check_output(
            ['nvpmodel', '-q'], stderr=subprocess.STDOUT)
        if not isinstance(output, str):
            output = output.decode('utf-8', 'replace')
    except OSError:
        print('%s Khong tim thay nvpmodel; neu day khong phai Jetson thi bo qua.' %
              WARN)
        return None
    except subprocess.CalledProcessError as exc:
        detail = exc.output.decode('utf-8', 'replace') \
            if not isinstance(exc.output, str) else exc.output
        print('%s Khong doc duoc power mode: %s' % (FAIL, detail.strip()))
        return False

    print('       nvpmodel -q: %s' % ' | '.join(
        line.strip() for line in output.splitlines() if line.strip()))
    if _is_5w_mode(output):
        print('%s Jetson dang o che do 5W.' % PASS)
        return True

    print('%s Jetson chua o 5W. Khong test servo khi camera/video dang tai CPU.' %
          FAIL)
    print('       Chay: sudo nvpmodel -m 1')
    print('       Sau do kiem tra lai: nvpmodel -q')
    return False


def check_config():
    step(3, 'Config')
    from jetracer_baseline.config import load_config
    path = os.path.join(ROOT, 'configs', 'default.yaml')
    try:
        cfg = load_config(path)
    except Exception as exc:
        print('%s Khong nap duoc %s: %s' % (FAIL, path, exc))
        return None
    print('%s %s' % (PASS, path))
    print('       v_max=%s  control_hz=%s  detect_hz=%s  signs.backend=%s'
          % (cfg.get('control.v_max'), cfg.get('pipeline.control_hz'),
             cfg.get('pipeline.detect_hz'), cfg.get('signs.backend')))
    return cfg


def check_camera(cfg, seconds, out_dir):
    step(4, 'Camera')
    import cv2
    from jetracer_baseline.camera import build_source, format_camera_environment

    print('       Moi truong: %s' % format_camera_environment())

    try:
        source = build_source(cfg, 'csi')
    except Exception as exc:
        print('%s Khong mo duoc camera: %s' % (FAIL, exc))
        print('       Kiem tra: cap CSI cam dung chieu? '
              'Thu: ls /dev/video*  hoac  nvgstcapture-1.0')
        return False

    print('       Backend ban dau: %s' % getattr(source, 'backend', 'khong ro'))

    n, t0, first = 0, time.time(), None
    read_error = None
    try:
        while (time.time() - t0) < seconds:
            ok, frame = source.read()
            if not ok:
                break
            if first is None:
                first = frame
            n += 1
    except Exception as exc:
        read_error = exc
    finally:
        source.release()

    print('       Backend su dung: %s' % getattr(source, 'backend', 'khong ro'))
    print('       Pipeline CSI: %s' % getattr(source, 'pipeline', 'khong ro'))
    for note in getattr(source, 'startup_notes', []):
        print('       Note: %s' % note)

    if read_error is not None:
        print('%s Camera open nhung doc frame bi loi: %s' % (FAIL, read_error))
        print('       CACH KHOI PHUC Failed to create CaptureSession:')
        print('       1) Jupyter: Kernel > Shut Down Kernel cho MOI notebook camera cu.')
        print('       2) Terminal: sudo systemctl restart nvargus-daemon')
        print('       3) Cho 2 giay, chay lai dung lenh check_hardware nay.')
        print('       Neu van loi: tat Jetson, rut nguon, cam lai cap CSI dung chieu.')
        return False

    if n == 0 or first is None:
        print('%s Mo duoc camera nhung khong doc duoc frame nao.' % FAIL)
        detail = getattr(source, 'last_error', None)
        if detail:
            print('       Chi tiet: %s' % detail)
        print('       Khoi phuc: dong moi kernel camera ->')
        print('         sudo systemctl restart nvargus-daemon')
        return False

    dt = time.time() - t0
    h, w = first.shape[:2]
    fps = n / dt if dt > 0 else 0.0
    print('%s Doc %d frame trong %.1f s -> %.1f FPS, do phan giai %dx%d'
          % (PASS, n, dt, fps, w, h))

    if fps < 25:
        print('%s Camera duoi 25 FPS. Vong dieu khien can >= 20 FPS de an 10 diem '
              '(DB muc 3.6) - camera cham la nut that dau tien.' % WARN)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    path = os.path.join(out_dir, 'camera_sample.jpg')
    cv2.imwrite(path, first)
    print('       Anh mau: %s  <- MO RA XEM: camera co dung huong khong, '
          'co thay mat duong ngay truoc mui xe khong?' % path)
    return True


def check_driver(kind, cfg):
    step(5, 'Backend dieu khien')
    from jetracer_baseline.control import driver as driver_mod

    report = driver_mod.probe()
    for name in sorted(report):
        mark = PASS if report[name].startswith('OK') else '      '
        print('%s %-8s : %s' % (mark, name, report[name]))

    if kind == 'dryrun':
        print('')
        print('%s Dang o che do dryrun - khong gui lenh xuong phan cung.' % WARN)
        print('       Chon --driver theo backend bao OK o tren.')
        return False

    if not report.get(kind, '').startswith('OK'):
        print('')
        print('%s Backend "%s" khong kha dung tren may nay.' % (FAIL, kind))
        if kind == 'nvidia':
            print('       Kiem tra file:')
            print('         ls /home/jetson/jetracer/jetracer/nvidia_racecar.py')
            print('       Neu repo o noi khac:')
            print('         export JETRACER_NVIDIA_ROOT=/duong/dan/toi/jetracer')
        return False

    # Import duoc thu vien chua co nghia la I2C/PCA9685 dung. Khoi tao driver
    # that va ghi neutral de bat loi sai dia chi/quyen truy cap ngay o preflight.
    from jetracer_baseline.control.driver import build_driver
    drv = None
    try:
        drv = build_driver(kind, cfg)
        drv.stop()
    except Exception as exc:
        print('')
        print('%s Import duoc backend nhung khoi tao phan cung that bai: %s' %
              (FAIL, exc))
        return False
    finally:
        if drv is not None:
            try:
                drv.close()
            except Exception:
                pass
    print('%s Da khoi tao driver va ghi neutral (steering=0, throttle=0).' % PASS)
    return True


def _hold_command(driver, steering, throttle, seconds):
    """Lam moi lenh de watchdog khong tu dua lai ve giua trong bai test."""
    deadline = time.monotonic() + max(0.0, float(seconds))
    while time.monotonic() < deadline:
        driver.set(steering, throttle)
        time.sleep(0.05)


def check_actuator(kind, lifted, cfg, steering_only=False):
    step(6, 'Servo lai + dong co  [RUI RO]')

    if not lifted:
        print('%s Bo qua. Buoc nay lam banh xe QUAY.' % WARN)
        print('       Ke xe len gia cho banh khong cham dat, roi chay lai voi:')
        print('         --actuator --wheels-are-lifted')
        return None

    from jetracer_baseline.control.driver import build_driver

    try:
        drv = build_driver(kind, cfg)
    except Exception as exc:
        print('%s Khong khoi tao duoc driver "%s": %s' % (FAIL, kind, exc))
        return False

    try:
        # 0.35 * gain 0.65 = output 0.2275, tuong duong khoang
        # 1329-1671 us voi pulse range 750-2250. Day la buoc quan sat co khi,
        # khong phai bai tim goc lai cuc dai.
        print('  Quet goc lai LOW-POWER (dong co KHONG chay)...')
        for label, value in (('giua ', 0.0), ('TRAI ', -0.35), ('giua ', 0.0),
                             ('PHAI ', 0.35), ('giua ', 0.0)):
            print('    steering = %+.2f  (%s)  <- nhin banh truoc' % (value, label))
            _hold_command(drv, value, 0.0, 0.50)

        if steering_only:
            print('')
            print('  Da bo qua motor theo --steering-only.')
        else:
            print('')
            print('  Xung ga rat ngan (0.15 trong 0.6 s)...')
            _hold_command(drv, 0.0, 0.15, 0.60)
            drv.stop()
            print('    xong, da ve 0')
    except Exception as exc:
        print('%s Loi khi dieu khien: %s' % (FAIL, exc))
        return False
    finally:
        try:
            drv.close()
        except Exception:
            pass

    print('')
    print('%s Da gui lenh. TU KIEM TRA bang mat:' % PASS)
    print('       - Banh truoc co danh sang TRAI khi in "TRAI" khong?')
    print('         Neu nguoc -> dao dau `steering` trong driver.py.')
    print('       - Khi steering = 0 banh co thang khong?')
    print('         Neu lech -> chinh control.driver.steering_offset trong configs/default.yaml.')
    print('       - Banh sau co quay TIEN khong? Neu lui -> dao dau throttle.')
    return True


def calibrate_steering(kind, cfg):
    """Quet dan output servo tu giua ra hai bien de tim gioi han CO KHI that.

    `pulse_width_range` (750-2250 us) la bien AN TOAN VE DIEN da duoc BTC xac
    nhan (control.txt: rong hon --> servo cham chan, stall, sut nguon). Nhung
    `steering_output_min/max` (mac dinh +/-0.40) la mot lop khoa MEM, chot tam
    trong luc bring-up vi chua ai do goc bam cua that su truoc khi banh cham
    gam/hoc banh. Ham nay giup do dung con so do, thay vi doan mo.
    """
    step(7, 'HIEU CHUAN GOC LAI TOI DA  [RUI RO - PHAI NHIN BANH LIEN TUC]')
    print('  Quet CHAM tu giua ra tung ben. NHIN THAT KY banh truoc.')
    print('  BAM Ctrl+C NGAY KHI banh VUA CHAM gam/hoc banh - dung doi den khi')
    print('  ket cung hoac nghe tieng servo ru. Dong "output=" IN NGAY TRUOC')
    print('  dong cuoi cung (truoc khi Ctrl+C) la gia tri con AN TOAN.')
    print('')

    from jetracer_baseline.control.driver import build_driver

    gain = float(cfg.get('control.driver.steering_gain', -0.65) or -0.65)
    offset = float(cfg.get('control.driver.steering_offset', 0.0) or 0.0)
    pulse_range = cfg.get('control.driver.pulse_width_range', [750, 2250]) \
        or [750, 2250]
    pulse_low, pulse_high = float(pulse_range[0]), float(pulse_range[1])

    try:
        drv = build_driver(kind, cfg)
    except Exception as exc:
        print('%s Khong khoi tao duoc driver "%s": %s' % (FAIL, kind, exc))
        return False

    # Mo het khoa mem de quet toi han cua pulse_width_range (van AN TOAN VE
    # DIEN); buoc nay chi di tim gioi han CO KHI nam ben trong bien do.
    drv.steering_output_min = -1.0
    drv.steering_output_max = 1.0

    step_size = 0.05
    command, sign = 0.0, 1.0
    try:
        for side, sign in (('PHAI', 1.0), ('TRAI', -1.0)):
            print('  -- Huong %s --' % side)
            _hold_command(drv, 0.0, 0.0, 0.3)
            command = 0.0
            while abs(command) < 1.0 - 1e-9:
                command = min(1.0, abs(command) + step_size) * sign
                output = command * gain + offset
                pulse = pulse_low + (output + 1.0) / 2.0 * (pulse_high - pulse_low)
                print('    lenh=%+.2f  ->  output=%+.3f  (~%d us)' % (
                    command, output, pulse))
                _hold_command(drv, command, 0.0, 0.7)
            _hold_command(drv, 0.0, 0.0, 0.3)
    except KeyboardInterrupt:
        prev_command = command - step_size * sign if abs(command) > step_size \
            else 0.0
        prev_output = prev_command * gain + offset
        print('')
        print('  Da dung theo yeu cau (Ctrl+C).')
        print('  Buoc VUA IN co the da qua gioi han co khi. Buoc TRUOC DO '
              '(an toan hon): output=%+.3f' % prev_output)
    finally:
        try:
            drv.set(0.0, 0.0)
        except Exception:
            pass
        try:
            drv.close()
        except Exception:
            pass

    print('')
    print('%s Hieu chuan xong (hoac da dung som theo Ctrl+C).' % PASS)
    print('       Lay gia tri output AN TOAN nho nhat trong hai ben (tru them')
    print('       ~0.03-0.05 du phong), roi dien doi xung vao configs/default.yaml:')
    print('         control.driver.steering_output_max:  <gia tri do duoc>')
    print('         control.driver.steering_output_min:  -<gia tri do duoc>')
    print('       Chua sua file config o day - ban tu chot sau khi da do that.')
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description='Bring-up phan cung JetRacer')
    parser.add_argument('--driver', choices=['dryrun', 'nvidia', 'ros'],
                        default='dryrun')
    parser.add_argument('--camera-seconds', type=float, default=3.0)
    parser.add_argument('--out', default='reports')
    parser.add_argument('--skip-camera', action='store_true')
    parser.add_argument('--actuator', action='store_true',
                        help='Chay buoc 5 (servo + dong co)')
    parser.add_argument('--wheels-are-lifted', action='store_true',
                        help='Xac nhan banh xe KHONG cham dat')
    parser.add_argument('--steering-only', action='store_true',
                        help='Chi test servo lai, khong quay motor')
    parser.add_argument('--calibrate-steering', action='store_true',
                        help='Quet dan output servo de tim gioi han co khi that '
                             '(can --wheels-are-lifted, doc lap voi --actuator)')
    args = parser.parse_args(argv)

    print('=' * 66)
    print('BRING-UP JETRACER - baseline Jetson AI Racer Challenge 2026')
    print('=' * 66)

    results = {}
    results['python'] = check_python()
    if not results['python']:
        print('\nDung lai: thieu thu vien co ban.')
        return 1

    results['power'] = check_power_mode()

    cfg = check_config()
    results['config'] = cfg is not None
    if cfg is None:
        return 1

    if args.skip_camera:
        results['camera'] = None
        print('\n(bo qua buoc camera theo yeu cau)')
    else:
        results['camera'] = check_camera(cfg, args.camera_seconds, args.out)

    results['driver'] = check_driver(args.driver, cfg)

    if args.actuator:
        if results['power'] is False:
            print('\n%s Bo qua actuator vi Jetson chua o power mode 5W.' % FAIL)
            results['actuator'] = False
        else:
            results['actuator'] = check_actuator(
                args.driver, args.wheels_are_lifted, cfg,
                steering_only=args.steering_only)
    else:
        results['actuator'] = None

    if args.calibrate_steering:
        if not args.wheels_are_lifted:
            print('\n%s --calibrate-steering can --wheels-are-lifted.' % FAIL)
            results['calibrate_steering'] = False
        elif results['power'] is False:
            print('\n%s Bo qua hieu chuan vi Jetson chua o power mode 5W.' % FAIL)
            results['calibrate_steering'] = False
        else:
            results['calibrate_steering'] = calibrate_steering(args.driver, cfg)
    else:
        results['calibrate_steering'] = None

    print('')
    print('=' * 66)
    print('TOM TAT')
    print('=' * 66)
    for name in ('python', 'power', 'config', 'camera', 'driver', 'actuator',
                 'calibrate_steering'):
        value = results.get(name)
        mark = 'chua chay' if value is None else ('OK' if value else 'LOI')
        print('  %-9s : %s' % (name, mark))

    blocking = [k for k, v in results.items() if v is False]
    if blocking:
        print('')
        print('Can xu ly: %s' % ', '.join(sorted(blocking)))
        return 1

    print('')
    print('Buoc tiep theo: tune nguong lane bang')
    print('  python3 tools/tune_lane.py --source csi --frames 5 --out reports/lane')
    return 0


if __name__ == '__main__':
    sys.exit(main())
