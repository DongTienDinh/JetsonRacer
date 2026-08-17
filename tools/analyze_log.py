# -*- coding: utf-8 -*-
"""Phan tich log CSV -> bang so lieu + bieu do cho Technical Paper.

    python tools/analyze_log.py logs/run_speed_*.csv --out reports/

Quy tac cua doi (docs/ke-hoach-thuc-nghiem.md §5): MOI con so trong paper phai
truy nguoc duoc ve mot file log cu the. Cong cu nay la duong duy nhat de sinh so.
Khong go tay con so nao vao paper.

The le §9.2 cam "so lieu khong co that"; §9.1 cho BTC quyen doi log de xac minh.
"""

from __future__ import print_function

import argparse
import csv
import glob
import io
import os
import sys


def _f(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_log(path):
    rows = []
    with io.open(path, 'r', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return ordered[idx]


def summarise(rows):
    cte = [abs(_f(r['cte'])) for r in rows if r.get('cte') not in (None, '')]
    fps = [_f(r['fps']) for r in rows if _f(r['fps']) > 0]
    lat = [_f(r['latency_ms']) for r in rows if _f(r['latency_ms']) > 0]
    sign_lat = [_f(r['sign_latency_ms']) for r in rows
                if r.get('sign_latency_ms') not in (None, '')]
    t_rel = [_f(r['t_rel']) for r in rows]

    lane_lost = sum(1 for r in rows if r.get('event') == 'lane_lost')
    events = {}
    for r in rows:
        ev = (r.get('event') or '').strip()
        if ev:
            events[ev] = events.get(ev, 0) + 1

    rms = 0.0
    if cte:
        rms = (sum(v * v for v in cte) / len(cte)) ** 0.5

    return {
        'frames': len(rows),
        'duration_s': (max(t_rel) - min(t_rel)) if t_rel else 0.0,
        'cte_rms': rms,
        'cte_p95': percentile(cte, 0.95),
        'fps_mean': (sum(fps) / len(fps)) if fps else 0.0,
        'fps_p05': percentile(fps, 0.05),
        'fps_min': min(fps) if fps else 0.0,
        'latency_p50': percentile(lat, 0.50),
        'latency_p95': percentile(lat, 0.95),
        'sign_latency_p95': percentile(sign_lat, 0.95),
        'sign_latency_max': max(sign_lat) if sign_lat else 0.0,
        'lane_lost_frames': lane_lost,
        'lane_lost_rate': (float(lane_lost) / len(rows)) if rows else 0.0,
        'events': events,
    }


# --------------------------------------------------------------------- scoring
def score_speed_track(duration_s, checkpoints, fps_mean, obstacle_hits,
                      lane_departures, false_starts=0, cp_points=10):
    """Cong thuc DB §3.6.

    cp_points: 10 theo §3.6, nhung vi du §8 lai ngu y 30. Chua duoc BTC xac nhan
    (xem BASELINE.md §6.1) -> de tham so hoa thay vi hardcode mot phia.
    """
    penalty_time = obstacle_hits * 10 + lane_departures * 15 + false_starts * 10
    total_time = duration_s + penalty_time

    cp_score = min(checkpoints * cp_points, cp_points * 3)
    fps_score = 10.0 if fps_mean >= 20.0 else 0.0
    time_score = max(0.0, 60.0 - 2.0 * (total_time / 10.0))
    penalty = obstacle_hits * 5 + lane_departures * 10 + false_starts * 10

    return {
        'checkpoint': cp_score,
        'fps': fps_score,
        'time': time_score,
        'penalty': penalty,
        'total_time_s': total_time,
        'score': max(0.0, cp_score + fps_score + time_score - penalty),
    }


def score_smart_city(duration_s, signs_read, signs_total, missed_stops=0,
                     ran_red=False, wrong_route=False):
    """Cong thuc DB §4.7.

    Cong thuc goc ghi `so bien doc duoc * (tong so bien / 70)` - sai don vi
    (xem BASELINE.md §6.2). O day dung dang duoc suy ra tu vi du §8:
        (so bien doc duoc / tong so bien) * 70
    """
    if ran_red:
        return {'sign': 0.0, 'time': 0.0, 'penalty': 0.0, 'score': 0.0,
                'note': 'HUY LUOT - vuot den do (DB muc 4.8)'}

    sign_score = 0.0
    if signs_total > 0:
        sign_score = min(70.0, (float(signs_read) / signs_total) * 70.0)
    time_score = max(0.0, 30.0 - 1.0 * (duration_s / 10.0))
    penalty = missed_stops * 10.0

    return {
        'sign': sign_score,
        'time': time_score,
        'penalty': penalty,
        'score': max(0.0, sign_score + time_score - penalty),
        'note': 'Ket thuc som - di sai lo trinh' if wrong_route else '',
    }


def final_score(speed_score, smart_score, paper=0.0, defense=0.0):
    """Thang §7.4: ST 30% + SC 40% + Paper 20% + Defense 10%."""
    return {
        'speed_weighted': speed_score * 0.30,
        'smart_weighted': smart_score * 0.40,
        'paper_weighted': paper * 0.20,
        'defense_weighted': defense * 0.10,
        'total': (speed_score * 0.30 + smart_score * 0.40
                  + paper * 0.20 + defense * 0.10),
    }


# ----------------------------------------------------------------------- plots
def make_plots(rows, out_dir, stem):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  (bo qua bieu do: chua cai matplotlib)')
        return []

    t = [_f(r['t_rel']) for r in rows]
    cte = [_f(r['cte']) for r in rows]
    fps = [_f(r['fps']) for r in rows]
    lat = [_f(r['latency_ms']) for r in rows]
    made = []

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(t, cte, linewidth=0.9)
    axes[0].axhline(0, color='grey', linewidth=0.6)
    axes[0].set_ylabel('cross-track error')
    axes[0].set_title('Log run: ' + stem)
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, fps, linewidth=0.9, color='tab:green')
    axes[1].axhline(20, color='red', linestyle='--', linewidth=1.0,
                    label='nguong 20 FPS (BTC)')
    axes[1].set_ylabel('FPS')
    axes[1].legend(loc='lower right', fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, lat, linewidth=0.9, color='tab:orange')
    axes[2].set_ylabel('latency (ms)')
    axes[2].set_xlabel('t (s) tu luc xuat phat')
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, stem + '_timeline.png')
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(path)
    return made


def main(argv=None):
    parser = argparse.ArgumentParser(description='Phan tich log JetRacer baseline')
    parser.add_argument('logs', nargs='+', help='Duong dan hoac glob toi file CSV')
    parser.add_argument('--out', default=None, help='Thu muc xuat bieu do')
    parser.add_argument('--checkpoints', type=int, default=3)
    parser.add_argument('--cp-points', type=int, default=10,
                        choices=[10, 30], help='10 theo DB muc 3.6, 30 theo vi du muc 8')
    parser.add_argument('--obstacle-hits', type=int, default=0)
    parser.add_argument('--lane-departures', type=int, default=0)
    parser.add_argument('--signs-read', type=int, default=0)
    parser.add_argument('--signs-total', type=int, default=0)
    parser.add_argument('--missed-stops', type=int, default=0)
    parser.add_argument('--ran-red', action='store_true')
    args = parser.parse_args(argv)

    paths = []
    for pattern in args.logs:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched if matched else [pattern])

    if args.out and not os.path.isdir(args.out):
        os.makedirs(args.out)

    any_ok = False
    for path in paths:
        if not os.path.exists(path):
            print('Bo qua (khong ton tai): %s' % path)
            continue
        rows = read_log(path)
        if not rows:
            print('Bo qua (rong): %s' % path)
            continue
        any_ok = True
        stem = os.path.splitext(os.path.basename(path))[0]
        s = summarise(rows)

        print('')
        print('=' * 68)
        print(path)
        print('=' * 68)
        print('  frames              : %d' % s['frames'])
        print('  duration            : %.2f s' % s['duration_s'])
        print('  CTE rms / p95       : %.4f / %.4f' % (s['cte_rms'], s['cte_p95']))
        print('  FPS mean / p05 / min: %.2f / %.2f / %.2f   [nguong BTC: 20]'
              % (s['fps_mean'], s['fps_p05'], s['fps_min']))
        print('  latency p50 / p95   : %.2f / %.2f ms' % (s['latency_p50'],
                                                          s['latency_p95']))
        print('  sign latency p95/max: %.2f / %.2f ms  [nguong BTC: 300]'
              % (s['sign_latency_p95'], s['sign_latency_max']))
        print('  mat lane            : %d frame (%.2f%%)'
              % (s['lane_lost_frames'], 100.0 * s['lane_lost_rate']))
        if s['events']:
            print('  events              : %s' % ', '.join(
                ['%s=%d' % (k, v) for k, v in sorted(s['events'].items())]))

        if 'smartcity' in stem:
            sc = score_smart_city(s['duration_s'], args.signs_read,
                                  args.signs_total, args.missed_stops,
                                  args.ran_red)
            print('  --- diem mo phong Smart City (DB muc 4.7) ---')
            print('    bien bao=%.2f  thoi gian=%.2f  penalty=%.2f  => %.2f/100'
                  % (sc['sign'], sc['time'], sc['penalty'], sc['score']))
            if sc.get('note'):
                print('    %s' % sc['note'])
        else:
            st = score_speed_track(s['duration_s'], args.checkpoints,
                                   s['fps_mean'], args.obstacle_hits,
                                   args.lane_departures, cp_points=args.cp_points)
            print('  --- diem mo phong Speed Track (DB muc 3.6) ---')
            print('    checkpoint=%.0f  fps=%.0f  thoi gian=%.2f  penalty=%.0f'
                  % (st['checkpoint'], st['fps'], st['time'], st['penalty']))
            print('    thoi gian ke phat=%.1f s  => %.2f/100'
                  % (st['total_time_s'], st['score']))

        if args.out:
            for made in make_plots(rows, args.out, stem):
                print('  bieu do             : %s' % made)

    return 0 if any_ok else 1


if __name__ == '__main__':
    sys.exit(main())
