# -*- coding: utf-8 -*-
"""Thu anh tu camera xe de gan nhan.

    # Thu anh roi rac moi 0.2 s, gan session theo dieu kien sang:
    python3 tools/collect_dataset.py --out data/raw --session chieu_nang --every 0.2

    # Quay video ca vong de lam bo replay offline:
    python3 tools/collect_dataset.py --out data/video --mode video --seconds 120

DB muc 7 noi ro: dataset BTC "khong hoan toan giong voi sa ban thuc te", nen day
la cong cu bat buoc dung, khong phai tuy chon.

Ten file mang san `session` -> khi chia train/val/test PHAI chia THEO SESSION,
khong random. Random split tren cac frame video lien tiep lam mAP val cao gia
tao (cac frame gan nhu giong het nhau nam ca o train lan val). Day dung la loai
loi giam khao se hoi o Oral Defense.
"""

from __future__ import print_function

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import cv2  # noqa: E402

from jetracer_baseline.camera import (  # noqa: E402
    _build_raw_source, build_source, shading_applied_at_source)
from jetracer_baseline.config import load_config   # noqa: E402


def collect_frames(source, out_dir, session, every, limit, show):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    saved = 0
    t_last = 0.0
    print('Dang thu... Ctrl-C de dung.')
    try:
        while limit is None or saved < limit:
            ok, frame = source.read()
            if not ok:
                print('Het nguon anh.')
                break
            now = time.time()
            if (now - t_last) < every:
                continue
            t_last = now

            name = '%s_%s_%04d.jpg' % (session, time.strftime('%Y%m%d_%H%M%S'), saved)
            cv2.imwrite(os.path.join(out_dir, name), frame)
            saved += 1
            if saved % 25 == 0:
                print('  da luu %d anh' % saved)
            if show:
                cv2.imshow('collect', frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
    except KeyboardInterrupt:
        print('\nDung theo yeu cau.')
    return saved


def record_video(source, out_dir, session, seconds, fps, show):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    ok, frame = source.read()
    if not ok:
        print('Khong doc duoc frame dau tien.')
        return None
    h, w = frame.shape[:2]

    path = os.path.join(out_dir, '%s_%s.mp4' % (session, time.strftime('%Y%m%d_%H%M%S')))
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not writer.isOpened():
        print('Khong mo duoc VideoWriter: ' + path)
        return None

    t0 = time.time()
    n = 0
    print('Dang quay vao %s ... Ctrl-C de dung.' % path)
    try:
        while (time.time() - t0) < seconds:
            ok, frame = source.read()
            if not ok:
                break
            writer.write(frame)
            n += 1
            if show:
                cv2.imshow('record', frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
    except KeyboardInterrupt:
        print('\nDung theo yeu cau.')
    finally:
        writer.release()

    print('Da quay %d frame (%.1f s) -> %s' % (n, time.time() - t0, path))
    return path



def _pick_source(args, cfg, kind, video_path=None, n_frames=None):
    """Chon nguon co / khong co sua mau.

    Mac dinh camera.shading.apply_at = source nen build_source() tra ve frame DA
    sua. Khi quay flat-field de hieu chuan lai shading thi phai lay anh THO,
    neu khong ta se do lens shading cua mot camera da duoc bu shading - ket qua
    la he so gan nhu vo hieu.
    """
    if args.raw:
        print('  --raw: lay anh THO, KHONG sua mau shading.')
        return _build_raw_source(cfg, kind, video_path, n_frames)
    if shading_applied_at_source(cfg):
        print('  Anh se duoc SUA MAU tai nguon (camera.shading.apply_at=source).')
        print('  Dung --raw neu dang quay flat-field de hieu chuan shading.')
    return build_source(cfg, kind, video_path=video_path, n_frames=n_frames)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Thu du lieu tu camera JetRacer')
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--out', default='data/raw')
    parser.add_argument('--mode', choices=['frames', 'video'], default='frames')
    parser.add_argument('--source', choices=['csi', 'video', 'synthetic'], default='csi')
    parser.add_argument('--video', default=None, help='Duong dan video khi --source video')
    parser.add_argument('--session', required=True,
                        help='Nhan dieu kien thu, vd: sang_som / den_phong / nguoc_sang. '
                             'DUNG de chia train/val/test theo session.')
    parser.add_argument('--every', type=float, default=0.2,
                        help='Giay giua hai anh (mode=frames)')
    parser.add_argument('--limit', type=int, default=None, help='So anh toi da')
    parser.add_argument('--seconds', type=float, default=120.0, help='mode=video')
    parser.add_argument('--fps', type=float, default=20.0, help='mode=video')
    parser.add_argument('--show', action='store_true', help='Hien cua so xem truoc')
    parser.add_argument(
        '--raw', action='store_true',
        help='Thu anh THO, BO QUA sua mau shading. BAT BUOC khi quay '
             'flat-field de hieu chuan shading - hieu chuan tren anh da sua '
             'se ra he so vo nghia.')
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    source = _pick_source(args, cfg, args.source, video_path=args.video)

    try:
        if args.mode == 'video':
            record_video(source, args.out, args.session, args.seconds,
                         args.fps, args.show)
        else:
            n = collect_frames(source, args.out, args.session, args.every,
                               args.limit, args.show)
            print('Tong: %d anh -> %s' % (n, args.out))
    finally:
        source.release()
        if args.show:
            cv2.destroyAllWindows()

    print('')
    print('Nho: chia train/val/test THEO SESSION, khong random split.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
