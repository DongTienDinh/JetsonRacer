# -*- coding: utf-8 -*-
"""Phan biet Jetson reboot/mat nguon voi kernel Python bi chet.

Mo mot Terminal rieng tren Jetson va chay truoc khi thu servo:

    python3 tools/diagnose_shutdown.py monitor

Neu xe tat/reboot, khoi dong lai roi chay:

    python3 tools/diagnose_shutdown.py check

File marker va heartbeat duoc flush lien tuc, nen van dung duoc sau mot lan
mat nguon dot ngot. Script chi doc thong tin he thong, khong dieu khien xe.
Python 3.6 compatible.
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MARKER = os.path.join(ROOT, 'reports', 'shutdown_probe.json')


def _read_text(path, default='?'):
    try:
        with open(path, 'r') as fh:
            return fh.read().strip()
    except (IOError, OSError):
        return default


def _boot_id():
    return _read_text('/proc/sys/kernel/random/boot_id')


def _uptime_s():
    text = _read_text('/proc/uptime', '0')
    try:
        return float(text.split()[0])
    except (IndexError, TypeError, ValueError):
        return 0.0


def _nvpmodel():
    try:
        output = subprocess.check_output(
            ['nvpmodel', '-q'], stderr=subprocess.STDOUT)
        if not isinstance(output, str):
            output = output.decode('utf-8', 'replace')
        return ' | '.join(line.strip() for line in output.splitlines()
                          if line.strip())
    except (OSError, subprocess.CalledProcessError):
        return 'khong-doc-duoc'


def _max_temperature_c():
    base = '/sys/class/thermal'
    values = []
    try:
        names = os.listdir(base)
    except OSError:
        return None
    for name in names:
        if not name.startswith('thermal_zone'):
            continue
        raw = _read_text(os.path.join(base, name, 'temp'), '')
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > 1000.0:
            value /= 1000.0
        values.append(value)
    return max(values) if values else None


def _ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(parent):
        os.makedirs(parent)


def _write_marker(path):
    _ensure_parent(path)
    marker = {
        'boot_id': _boot_id(),
        'created_unix': time.time(),
        'uptime_s': _uptime_s(),
        'platform': platform.platform(),
        'nvpmodel': _nvpmodel(),
    }
    with open(path, 'w') as fh:
        json.dump(marker, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    return marker


def monitor(path, interval_s):
    marker = _write_marker(path)
    heartbeat_path = os.path.splitext(path)[0] + '.heartbeat.csv'
    print('Marker   : %s' % path)
    print('Heartbeat: %s' % heartbeat_path)
    print('boot_id  : %s' % marker['boot_id'])
    print('nvpmodel : %s' % marker['nvpmodel'])
    print('Hay GIU terminal nay mo, roi sang Jupyter thu be lai.')
    print('Neu khong mat nguon, bam Ctrl+C de dung.')

    _ensure_parent(heartbeat_path)
    with open(heartbeat_path, 'w') as fh:
        writer = csv.writer(fh)
        writer.writerow(['timestamp_unix', 'uptime_s', 'load_1m',
                         'max_temperature_c'])
        fh.flush()
        os.fsync(fh.fileno())
        try:
            while True:
                try:
                    load_1m = os.getloadavg()[0]
                except (AttributeError, OSError):
                    load_1m = 0.0
                temp = _max_temperature_c()
                writer.writerow([
                    '%.6f' % time.time(),
                    '%.3f' % _uptime_s(),
                    '%.3f' % load_1m,
                    '' if temp is None else '%.2f' % temp,
                ])
                fh.flush()
                os.fsync(fh.fileno())
                time.sleep(max(0.10, float(interval_s)))
        except KeyboardInterrupt:
            print('\nDa dung monitor; bay gio chay lenh `check`.')
    return 0


def check(path):
    try:
        with open(path, 'r') as fh:
            marker = json.load(fh)
    except (IOError, OSError, ValueError) as exc:
        print('[FAIL] Khong doc duoc marker %s: %s' % (path, exc))
        print('Hay chay `monitor` truoc khi thu servo.')
        return 2

    old_boot = marker.get('boot_id', '?')
    new_boot = _boot_id()
    print('boot_id luc bat dau: %s' % old_boot)
    print('boot_id hien tai  : %s' % new_boot)
    if old_boot != '?' and new_boot != '?' and old_boot != new_boot:
        print('[REBOOT CONFIRMED] Jetson da khoi dong lai.')
        print('Day khong phai chi la exception Python/Jupyter. Uu tien kiem tra '
              'pin, rail 5V, servo ket co khi va power mode 5W.')
        return 1

    print('[NO REBOOT] Boot ID khong doi.')
    print('Neu giao dien van mat, loi nam o process/kernel/camera/I2C; lay traceback '
          'Jupyter va `journalctl -b -n 100 --no-pager`.')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Chan doan JetRacer reboot hay chi chet process Python')
    parser.add_argument('command', choices=['monitor', 'check'])
    parser.add_argument('--out', default=DEFAULT_MARKER)
    parser.add_argument('--interval', type=float, default=0.25)
    args = parser.parse_args(argv)
    if args.command == 'monitor':
        return monitor(args.out, args.interval)
    return check(args.out)


if __name__ == '__main__':
    sys.exit(main())
