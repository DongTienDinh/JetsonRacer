# -*- coding: utf-8 -*-
"""Bring-up xe theo tung buoc, tu an toan den rui ro.

    python3 tools/check_hardware.py                  # buoc 1-4, KHONG chay motor
    python3 tools/check_hardware.py --driver nvidia --actuator --wheels-are-lifted

Thu tu cac buoc co chu dich: moi buoc chi chay khi buoc truoc da qua. Buoc 5
(dong co) bi khoa sau co --wheels-are-lifted vi xe co the lao di.

!! TRUOC KHI CHAY BUOC 5: KE XE LEN GIA, BANH KHONG CHAM DAT !!
"""

from __future__ import print_function

import argparse
import os
import platform
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


def check_config():
    step(2, 'Config')
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
    step(3, 'Camera')
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
    for note in getattr(source, 'startup_notes', []):
        print('       Note: %s' % note)

    if read_error is not None:
        print('%s Camera open nhung doc frame bi loi: %s' % (FAIL, read_error))
        return False

    if n == 0 or first is None:
        print('%s Mo duoc camera nhung khong doc duoc frame nao.' % FAIL)
        detail = getattr(source, 'last_error', None)
        if detail:
            print('       Chi tiet: %s' % detail)
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


def check_driver(kind):
    step(4, 'Backend dieu khien')
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
        return False
    return True


def check_actuator(kind, lifted, cfg):
    step(5, 'Servo lai + dong co  [RUI RO]')

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
        print('  Quet goc lai (dong co KHONG chay)...')
        for label, value in (('giua ', 0.0), ('TRAI ', -0.6), ('giua ', 0.0),
                             ('PHAI ', 0.6), ('giua ', 0.0)):
            drv.set(value, 0.0)
            print('    steering = %+.2f  (%s)  <- nhin banh truoc' % (value, label))
            time.sleep(1.2)

        print('')
        print('  Xung ga rat ngan (0.15 trong 0.6 s)...')
        drv.set(0.0, 0.15)
        time.sleep(0.6)
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
    args = parser.parse_args(argv)

    print('=' * 66)
    print('BRING-UP JETRACER - baseline Jetson AI Racer Challenge 2026')
    print('=' * 66)

    results = {}
    results['python'] = check_python()
    if not results['python']:
        print('\nDung lai: thieu thu vien co ban.')
        return 1

    cfg = check_config()
    results['config'] = cfg is not None
    if cfg is None:
        return 1

    if args.skip_camera:
        results['camera'] = None
        print('\n(bo qua buoc camera theo yeu cau)')
    else:
        results['camera'] = check_camera(cfg, args.camera_seconds, args.out)

    results['driver'] = check_driver(args.driver)

    if args.actuator:
        results['actuator'] = check_actuator(args.driver, args.wheels_are_lifted, cfg)
    else:
        results['actuator'] = None

    print('')
    print('=' * 66)
    print('TOM TAT')
    print('=' * 66)
    for name in ('python', 'config', 'camera', 'driver', 'actuator'):
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
