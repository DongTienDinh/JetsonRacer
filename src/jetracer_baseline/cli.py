# -*- coding: utf-8 -*-
"""Entry point.

    # Smoke test tren laptop, khong can xe, khong can model:
    python -m src.jetracer_baseline.cli replay --source synthetic --frames 200

    # Replay video da quay tren sa ban:
    python -m src.jetracer_baseline.cli replay --source video --video data/lap1.mp4

    # Tren xe (backend da chot la `nvidia` theo control.txt cua BTC; van nen
    # chay `probe` ngay dau co xe de xac nhan thu vien co san):
    python3 -m src.jetracer_baseline.cli run --task smartcity --driver nvidia

    # Ngay thi, luot 2-3 Speed Track:
    python3 -m src.jetracer_baseline.cli run --task speed --driver nvidia \
        --override configs/fast.yaml
"""

import argparse
import os
import sys

# Cho phep chay bang `python src/jetracer_baseline/cli.py` lan `python -m ...`
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from jetracer_baseline.camera import build_source
    from jetracer_baseline.config import load_config
    from jetracer_baseline.control import driver as driver_mod
    from jetracer_baseline.pipeline import Runner
else:
    from .camera import build_source
    from .config import load_config
    from .control import driver as driver_mod
    from .pipeline import Runner


def _add_common(parser):
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--override', action='append', default=[],
                        help='File config ghi de, co the lap lai')
    parser.add_argument('--task', choices=['speed', 'smartcity'], default='speed')
    parser.add_argument('--log-dir', default=None)
    parser.add_argument('--max-seconds', type=float, default=300.0,
                        help='Mac dinh 300 = gioi han 5 phut/luot cua BTC')
    parser.add_argument('--record', action='store_true',
                        help='Ghi video luc chay, cung thu muc va run_name voi CSV')
    parser.add_argument('--record-every', type=int, default=1,
                        help='Chi ghi 1/N frame (mac dinh: moi frame)')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='jetracer-baseline')
    sub = parser.add_subparsers(dest='cmd')

    p_run = sub.add_parser('run', help='Chay tren xe that')
    _add_common(p_run)
    p_run.add_argument(
        '--driver', choices=['dryrun', 'nvidia', 'ros'], required=True,
        help='Bat buoc chon ro; dung nvidia cho xe that, dryrun chi mo phong')

    p_replay = sub.add_parser('replay', help='Chay offline (video hoac anh tong hop)')
    _add_common(p_replay)
    p_replay.add_argument('--source', choices=['synthetic', 'video'], default='synthetic')
    p_replay.add_argument('--video', default=None)
    p_replay.add_argument('--frames', type=int, default=300)
    p_replay.add_argument('--realtime', action='store_true',
                          help='Giu nhip control_hz (mac dinh: chay het toc do)')

    sub.add_parser('probe', help='Kiem tra backend dieu khien nao kha dung')

    args = parser.parse_args(argv)

    if args.cmd == 'probe':
        print('Backend dieu khien kha dung tren may nay:')
        for name, status in sorted(driver_mod.probe().items()):
            print('  %-8s : %s' % (name, status))
        print('\n-> Chot lai `--driver` theo ket qua nay TRUOC khi lam viec khac.')
        return 0

    if args.cmd is None:
        parser.print_help()
        return 1

    cfg = load_config(args.config, args.override)
    log_dir = args.log_dir or cfg.get('logging.dir', 'logs')
    record_dir = log_dir if args.record else None

    if args.cmd == 'replay':
        source = build_source(cfg, args.source, video_path=args.video,
                              n_frames=args.frames)
        runner = Runner(cfg, task=args.task, driver_kind='dryrun', source=source,
                        log_dir=log_dir, sync=True, realtime=args.realtime,
                        record_dir=record_dir, record_every=args.record_every)
        runner.run(max_seconds=args.max_seconds, max_frames=args.frames)
        return 0

    source = build_source(cfg, 'csi')
    runner = Runner(cfg, task=args.task, driver_kind=args.driver, source=source,
                    log_dir=log_dir, record_dir=record_dir,
                    record_every=args.record_every)
    runner.run(max_seconds=args.max_seconds)
    return 0


if __name__ == '__main__':
    sys.exit(main())
