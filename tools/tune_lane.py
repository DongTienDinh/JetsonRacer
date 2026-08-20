# -*- coding: utf-8 -*-
"""Tune nguong bam lane bang cach nhin tung buoc xu ly.

    # Tren xe, dat len sa ban that:
    python3 tools/tune_lane.py --source csi --frames 5 --out reports/lane

    # Tren laptop, tu video da quay:
    python tools/tune_lane.py --source video --video data/lap1.mp4 --frames 8

    # Quet thu nhieu gia tri nguong cung luc.
    # CHU Y dau `=`: gia tri am bat dau bang `-` nen argparse se hieu nham la
    # ten option neu viet cach ra.
    python tools/tune_lane.py --source video --video data/lap1.mp4 \
        --sweep-c=-18,-12,-6 --frames 3

Xuat anh ghep 4 o cho moi frame:
    [1] anh goc + vung ROI + cte     [2] sau nhi phan hoa
    [3] sau khi cat ROI              [4] sau bird's-eye warp

Cach doc:
  - O [2] toan trang hoac toan den  -> chinh lane.threshold_c
  - O [3] khong thay mat duong      -> chinh lane.roi_top
  - O [4] hai mep duong khong gan song song -> chinh lane.warp_src
  - cte khong doi dau khi xe lech trai/phai -> warp hoac ROI sai

Day la cong cu dung nhieu nhat trong 5 phut chuan bi tren sa ban ngay thi:
chi can doi threshold_c la du bu phan lon thay doi anh sang.
"""

from __future__ import print_function

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from jetracer_baseline.camera import build_source  # noqa: E402
from jetracer_baseline.config import load_config   # noqa: E402
from jetracer_baseline.perception import build_lane_detector
from jetracer_baseline.camera import shading_applied_at_source
from jetracer_baseline.perception.shading import ShadingCorrector  # noqa: E402


def _label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _to_bgr(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def build_panel(dbg, cfg, title):
    res = dbg['result']
    h, w = dbg['small'].shape[:2]

    overlay = dbg['small'].copy()
    y0, y1 = dbg['roi_y']
    cv2.rectangle(overlay, (0, y0), (w - 1, y1 - 1), (0, 200, 255), 1)

    # Vach giua anh (muc tieu) va vi tri tam lane do duoc
    cv2.line(overlay, (w // 2, y0), (w // 2, h - 1), (120, 120, 120), 1)
    cx = int(w / 2.0 + res.cte * (w / 2.0))
    colour = (0, 220, 0) if res.found else (0, 0, 230)
    cv2.line(overlay, (cx, y0), (cx, h - 1), colour, 2)

    status = 'FOUND' if res.found else 'LOST'
    overlay = _label(overlay, '%s  cte=%+.3f  curv=%+.3f  px=%d'
                     % (status, res.cte, res.curvature, res.n_pixels))

    cells = [
        overlay,
        _label(_to_bgr(dbg['binary']), 'binary  c=%s block=%s'
               % (cfg.get('lane.threshold_c'), cfg.get('lane.threshold_block'))),
        _label(_to_bgr(dbg['masked']), 'ROI  roi_top=%s' % cfg.get('lane.roi_top')),
        _label(_to_bgr(dbg['warped']), 'birds-eye  (min_pixels=%s)'
               % cfg.get('lane.min_pixels')),
    ]

    top = np.hstack([cells[0], cells[1]])
    bottom = np.hstack([cells[2], cells[3]])
    panel = np.vstack([top, bottom])

    banner = np.zeros((22, panel.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner, title, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([banner, panel])


def run_once(cfg, source, frames, every, out_dir, tag):
    det = build_lane_detector(cfg)
    # PHAI sua mau truoc khi tune nguong, dung y het vong chay tren xe.
    # Tune tren anh chua sua se ra nguong bu lai cho am do o vien - roi khi chay
    # that (co sua mau) thi nguong do lai sai. Hai duong xu ly phai giong nhau.
    if shading_applied_at_source(cfg):
        # build_source() da boc ShadedSource -> frame vao day da sach roi.
        shading = ShadingCorrector.disabled()
    else:
        shading = ShadingCorrector.from_config(cfg)
    proc_size = (int(cfg.get('pipeline.proc_width', 320)),
                 int(cfg.get('pipeline.proc_height', 240)))
    if shading_applied_at_source(cfg):
        print('  shading: BAT tai NGUON camera (frame vao day da sach)')
    else:
        print('  shading: %s trong vong xu ly'
              % ('BAT' if shading.enabled else 'TAT (chua hieu chuan)'))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    saved, seen, found = [], 0, 0
    while len(saved) < frames:
        ok, frame = source.read()
        if not ok:
            break
        seen += 1
        if every > 1 and (seen % every) != 0:
            continue

        dbg = det.debug_process(shading.apply_resized(frame, proc_size))
        if dbg['result'].found:
            found += 1

        title = '%s  frame %d  cte=%+.3f' % (tag, seen, dbg['result'].cte)
        panel = build_panel(dbg, cfg, title)
        path = os.path.join(out_dir, '%s_%03d.png' % (tag, len(saved)))
        cv2.imwrite(path, panel)
        saved.append(path)
        print('  %s   cte=%+.3f  %s  px=%d'
              % (path, dbg['result'].cte,
                 'FOUND' if dbg['result'].found else 'LOST ',
                 dbg['result'].n_pixels))

    rate = (100.0 * found / len(saved)) if saved else 0.0
    print('  -> bat duoc lane o %d/%d anh (%.0f%%)' % (found, len(saved), rate))
    return rate


def main(argv=None):
    parser = argparse.ArgumentParser(description='Tune nguong bam lane')
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--override', action='append', default=[],
                        help='file config de trong len tren, co the lap lai. '
                             'Vd: --override configs/cnn.yaml de xem mask CNN')
    parser.add_argument('--source', choices=['csi', 'video', 'synthetic'],
                        default='csi')
    parser.add_argument('--video', default=None)
    parser.add_argument('--frames', type=int, default=5, help='So anh xuat ra')
    parser.add_argument('--every', type=int, default=1,
                        help='Lay 1 anh moi N frame (de trai deu ca vong)')
    parser.add_argument('--out', default='reports/lane')
    parser.add_argument('--sweep-c', default=None,
                        help='Quet lane.threshold_c, vd: --sweep-c=-18,-12,-6 '
                             '(phai dung dau = vi gia tri am)')
    parser.add_argument('--sweep-roi', default=None,
                        help='Quet lane.roi_top, vd: 0.45,0.55,0.65')
    args = parser.parse_args(argv)

    values = [(None, None)]
    if args.sweep_c:
        values = [('threshold_c', float(v)) for v in args.sweep_c.split(',')]
    elif args.sweep_roi:
        values = [('roi_top', float(v)) for v in args.sweep_roi.split(',')]

    scores = []
    for key, value in values:
        cfg = load_config(args.config, args.override)
        if key:
            cfg.set('lane.' + key, value)
            tag = '%s_%s' % (key, str(value).replace('.', 'p').replace('-', 'm'))
            print('')
            print('lane.%s = %s' % (key, value))
        else:
            tag = 'lane'

        source = build_source(cfg, args.source, video_path=args.video,
                              n_frames=args.frames * max(1, args.every))
        try:
            rate = run_once(cfg, source, args.frames, args.every, args.out, tag)
        finally:
            source.release()
        scores.append((tag, key, value, rate))

    if len(scores) > 1:
        print('')
        print('=== Ket qua quet ===')
        for tag, key, value, rate in sorted(scores, key=lambda s: -s[3]):
            print('  lane.%-12s = %-8s  bat lane %.0f%%' % (key, value, rate))
        best = max(scores, key=lambda s: s[3])
        print('')
        print('Tot nhat: lane.%s = %s' % (best[1], best[2]))
        print('LUU Y: ti le bat lane cao chua chac la tot - PHAI mo anh ra xem')
        print('       cte co doi dau dung khi xe lech trai/phai khong.')

    print('')
    print('Mo cac anh trong %s de doi chieu voi phan "Cach doc" o dau file nay.'
          % args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
