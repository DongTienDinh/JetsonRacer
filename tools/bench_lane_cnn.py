# -*- coding: utf-8 -*-
"""Do latency bam vach CNN TREN XE. Con so quyet dinh 10 diem FPS.

    # Tren Jetson, sau khi da build engine:
    python3 tools/bench_lane_cnn.py --engine models/lane_tiny.engine
    python3 tools/bench_lane_cnn.py --engine models/lane_tiny.engine --source csi

Diem FPS cua Speed Track la NHI PHAN: >= 20 FPS duoc 10 diem, 19.9 duoc 0. Nen
phai do trước, khong phai phat hien lúc chạy thi.

Do RIENG tung phan de biet cat o dau khi thieu:
    tien xu ly  : crop + resize + doi truc
    suy dien    : TensorRT
    hau xu ly   : warp + band + polyfit
Ngan sach ca vong o 20 FPS la 50 ms, va bam vach chi la MOT phan trong do -
con doc camera, stopline, FSM, driver, ghi log. Nham 20 ms cho ca ba phan tren.

Bao cao p50 va p95, khong bao trung binh: mot frame cham 300 ms van cho trung
binh dep trong khi xe da lech lane.
"""

from __future__ import print_function

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.jetracer_baseline.config import load_config          # noqa: E402
from src.jetracer_baseline.perception.lane_cnn import CnnLaneDetector  # noqa: E402


def frames_from(source, cfg, n, video):
    if source == 'csi' or source == 'video':
        from src.jetracer_baseline.camera import build_source, LatestFrameGrabber
        src = build_source(cfg, source, video_path=video)
        grab = LatestFrameGrabber(src)
        grab.start()
        out = []
        while len(out) < n:
            f = grab.latest()
            if f is not None:
                out.append(f[0] if isinstance(f, tuple) else f)
            time.sleep(0.005)
        grab.stop()
        return out
    rng = np.random.default_rng(0) if hasattr(np.random, 'default_rng') else None
    shape = (int(cfg.get('camera.height', 480)), int(cfg.get('camera.width', 640)), 3)
    if rng is not None:
        return [rng.integers(0, 255, shape, dtype=np.uint8) for _ in range(n)]
    return [np.random.randint(0, 255, shape).astype(np.uint8) for _ in range(n)]


def pct(a, q):
    return float(np.percentile(np.array(a), q))


def watch(det, cfg, args):
    """In `cte` lien tuc. KHONG gui lenh nao xuong phan cung.

    PHEP KIEM TRA QUAN TRONG NHAT TRUOC KHI CHO XE CHAY:
    cam xe day sang TRAI -> cte phai AM; day sang PHAI -> cte phai DUONG.
    Sai dau o day thi PID se lai xe ra khoi lane NHANH HON la vao. Quy trinh CV
    (docs/quy-trinh-test-xe.md buoc 5) da yeu cau dung phep thu nay - CNN cung
    phai qua, va no la mot mo hinh moi nen cang phai kiem lai.
    """
    from src.jetracer_baseline.camera import build_source, LatestFrameGrabber
    src = build_source(cfg, args.source if args.source != 'synthetic' else 'csi',
                       video_path=args.video)
    grab = LatestFrameGrabber(src)
    grab.start()
    print('\nDAY XE SANG TRAI -> cte am.  SANG PHAI -> cte duong.')
    print('Sai dau = DUNG LAI, dung chay tiep.   Ctrl-C de thoat.\n')
    print('%-7s %8s %8s %8s %7s %8s' %
          ('vach', 'cte', 'lookah', 'curv', 'band', 'infer ms'))
    try:
        while True:
            f = grab.latest()
            if f is None:
                time.sleep(0.02)
                continue
            frame = f[0] if isinstance(f, tuple) else f
            r = det.process(frame)
            arrow = '<<<' if r.cte < -0.08 else ('>>>' if r.cte > 0.08 else ' | ')
            print('%-7s %+8.3f %+8.3f %+8.3f %7d %8.2f  %s'
                  % ('CO' if r.found else 'MAT', r.cte, r.cte_lookahead,
                     r.curvature, r.n_bands, det.last_infer_ms, arrow))
            time.sleep(0.15)
    except KeyboardInterrupt:
        print('\nthoat')
    finally:
        grab.stop()
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--override', action='append', default=[])
    ap.add_argument('--engine', default='')
    ap.add_argument('--source', choices=['synthetic', 'csi', 'video'],
                    default='synthetic')
    ap.add_argument('--video', default=None)
    ap.add_argument('--frames', type=int, default=200)
    ap.add_argument('--watch', action='store_true',
                    help='in cte lien tuc de kiem tra DAU khi day xe trai/phai')
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    if args.engine:
        cfg.set('lane.cnn.engine', args.engine)
    engine = cfg.get('lane.cnn.engine', '')
    if not engine:
        print('Chua co lane.cnn.engine. Dung --engine <duong dan .engine>')
        return 2

    print('engine : %s' % engine)
    det = CnnLaneDetector(cfg)

    if args.watch:
        return watch(det, cfg, args)

    frames = frames_from(args.source, cfg, args.frames, args.video)
    print('nguon  : %s, %d frame %s' % (args.source, len(frames),
                                        frames[0].shape))

    pre, inf, post, tot, found = [], [], [], [], 0
    for f in frames:
        t0 = time.time()
        x = det.preprocess(f)
        t1 = time.time()
        det.engine.infer(x)
        t2 = time.time()
        res = det.process(f)              # do lai ca vong, gom hau xu ly
        t3 = time.time()
        pre.append((t1 - t0) * 1e3)
        inf.append((t2 - t1) * 1e3)
        post.append((t3 - t2) * 1e3 - det.last_infer_ms - (t1 - t0) * 1e3)
        tot.append(det.last_total_ms)
        found += 1 if res.found else 0

    print('\n%-12s %8s %8s' % ('phan', 'p50 ms', 'p95 ms'))
    print('-' * 30)
    for name, a in (('tien xu ly', pre), ('suy dien', inf),
                    ('hau xu ly', post), ('CA VONG', tot)):
        print('%-12s %8.2f %8.2f' % (name, pct(a, 50), pct(a, 95)))
    print('-' * 30)
    fps50 = 1000.0 / max(pct(tot, 50), 1e-6)
    fps95 = 1000.0 / max(pct(tot, 95), 1e-6)
    print('bam vach dat %.1f FPS (p50) / %.1f FPS (p95) NEU chay mot minh'
          % (fps50, fps95))
    print('tim thay vach: %d/%d frame' % (found, len(frames)))
    if det.n_disagree:
        print('seg va reg bat dong y: %d frame' % det.n_disagree)

    budget = 20.0
    p95 = pct(tot, 95)
    print('\nNgan sach nham cho bam vach: %.0f ms (p95). Do duoc: %.2f ms -> %s'
          % (budget, p95, 'CON DU' if p95 <= budget else 'VUOT'))
    print('Day KHONG phai FPS thi dau. Con doc camera, stopline, FSM, driver,')
    print('ghi log. So doi chieu moc 20 FPS chi lay tu mot luot chay that:')
    print('  python3 -m src.jetracer_baseline.cli run --task speed --driver nvidia')
    return 0 if p95 <= budget else 1


if __name__ == '__main__':
    sys.exit(main())
