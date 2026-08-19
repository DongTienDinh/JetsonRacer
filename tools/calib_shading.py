# -*- coding: utf-8 -*-
"""Hieu chuan lens shading mau cho camera CSI -> configs/shading.yaml.

HAI CHE DO, KHAC NHAU VE DO TIN CAY:

  --mode flatfield   (CHINH XAC - dung cai nay khi co the)
      Quay mot doan camera nhin vao MAT PHANG MOT MAU, SANG DEU, phu kin khung
      hinh: to giay A4 trang, tam foam trang, hoac tuong trang. Anh phai khong
      lay net vao chi tiet nao (de cach ~5-10 cm, hoi mo la tot).
      Moi lech mau do duoc luc do = 100% do ong kinh -> he so chinh xac.

  --mode drive       (TAM THOI - khi chua kip quay flat-field)
      Uoc luong tu mot video chay binh thuong, lay trung binh theo thoi gian.
      Noi dung canh KHONG trung binh het duoc (troi/tran luon o tren, san luon
      o duoi), nen he so co sai so. Kiem tra tinh doi xung o phan bao cao:
      neu trai/phai va tren/duoi lech nhau > 10% thi ket qua dang bi canh lam
      nhieu, dung tin - phai quay flat-field.

Vi du:
    python tools/calib_shading.py --mode flatfield --source flat.avi
    python tools/calib_shading.py --mode drive --source raw_camera.avi
    python tools/calib_shading.py --mode drive --source raw_camera.avi --preview out.png
"""

import argparse
import io
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'src'))

from jetracer_baseline.perception.shading import (  # noqa: E402
    ShadingCorrector, fit_coefficients, measure_ratio_profile)


def mean_frame(source, max_frames=None, stride=1):
    """Anh trung binh theo thoi gian. Nhan file video hoac thu muc anh."""
    if os.path.isdir(source):
        names = sorted(n for n in os.listdir(source)
                       if n.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
        if not names:
            raise IOError('Thu muc khong co anh: ' + source)
        acc, n = None, 0
        for i, name in enumerate(names):
            if i % stride:
                continue
            img = cv2.imread(os.path.join(source, name))
            if img is None:
                continue
            acc = img.astype(np.float64) if acc is None else acc + img
            n += 1
            if max_frames and n >= max_frames:
                break
        if n == 0:
            raise IOError('Khong doc duoc anh nao trong: ' + source)
        return acc / n, n

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError('Khong mo duoc nguon: ' + source)
    acc, n, i = None, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            acc = frame.astype(np.float64) if acc is None else acc + frame
            n += 1
            if max_frames and n >= max_frames:
                break
        i += 1
    cap.release()
    if n == 0:
        raise IOError('Nguon khong co frame nao: ' + source)
    return acc / n, n


def symmetry_report(mean_bgr):
    """Kiem tra R/G co doi xung khong. Doi xung => lens shading, khong phai canh."""
    b, g, r = (mean_bgr[:, :, i].astype(np.float64) for i in range(3))
    rg = r / np.maximum(g, 1e-6)
    h, w = rg.shape
    left, right = rg[:, :w // 6].mean(), rg[:, -w // 6:].mean()
    top, bottom = rg[:h // 6, :].mean(), rg[-h // 6:, :].mean()
    lr = abs(left - right) / max((left + right) / 2.0, 1e-6)
    tb = abs(top - bottom) / max((top + bottom) / 2.0, 1e-6)
    return {
        'left': left, 'right': right, 'top': top, 'bottom': bottom,
        'lr_mismatch': lr, 'tb_mismatch': tb,
        'ok': (lr < 0.10 and tb < 0.10),
    }


def clipping_report(mean_bgr, corrector):
    """Bao nhieu pixel bi cham tran 255 sau khi sua (tren anh trung binh)."""
    out = corrector.apply(np.clip(mean_bgr, 0, 255).astype(np.uint8))
    return float(np.mean(out >= 255))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['flatfield', 'drive'], required=True)
    ap.add_argument('--source', required=True,
                    help='File video hoac thu muc anh')
    ap.add_argument('--out', default='configs/shading.yaml')
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--bins', type=int, default=8)
    ap.add_argument('--preview', default=None,
                    help='Ghi anh so sanh truoc/sau ra duong dan nay')
    ap.add_argument('--force', action='store_true',
                    help='Van ghi file du kiem tra doi xung that bai')
    args = ap.parse_args(argv)

    mean, n = mean_frame(args.source, args.max_frames, max(1, args.stride))
    print('Nguon        : %s' % args.source)
    print('Che do       : %s' % args.mode)
    print('So frame gop : %d   kich thuoc %dx%d' % (n, mean.shape[1], mean.shape[0]))

    sym = symmetry_report(mean)
    print('\n--- Kiem tra doi xung R/G (lens shading phai doi xung) ---')
    print('  trai %.3f  |  phai %.3f   -> lech %.1f%%' % (
        sym['left'], sym['right'], 100 * sym['lr_mismatch']))
    print('  tren %.3f  |  duoi %.3f   -> lech %.1f%%' % (
        sym['top'], sym['bottom'], 100 * sym['tb_mismatch']))
    if sym['ok']:
        print('  => DOI XUNG. Do lech mau la do ong kinh, sua duoc.')
    else:
        print('  => KHONG DOI XUNG (>10%). Anh trung binh dang bi noi dung canh')
        print('     lam lech, HOAC camera that su bi che mot ben. He so fit ra')
        print('     se khong dung. Hay quay lai bang --mode flatfield.')

    profile = measure_ratio_profile(mean, n_bins=args.bins)
    print('\n--- Ti le do duoc theo ban kinh (truoc khi sua) ---')
    print('  %-12s %8s %8s %10s' % ('r^2', 'R/G', 'B/G', 'so pixel'))
    for p in profile:
        print('  %.2f-%.2f    %8.3f %8.3f %10d' % (
            p['r2_lo'], p['r2_hi'], p['rg'], p['bg'], p['n']))

    rg_now = [p['rg'] for p in profile]
    if (max(rg_now) - min(rg_now)) < 0.08:
        print('\nCANH BAO: bien do R/G da rat nho (%.3f). Video nay CO VE DA'
              ' DUOC SUA MAU roi.' % (max(rg_now) - min(rg_now)))
        print('  Neu camera.shading.apply_at = source thi moi video ghi ra deu'
              ' da sach; hieu chuan tiep tren no se ra he so gan nhu vo hieu.')
        print('  Hieu chuan lai phai quay flat-field MOI voi shading TAT'
              ' (camera.shading.enabled: false).')

    coeff_r, coeff_b = fit_coefficients(profile)
    corrector = ShadingCorrector(coeff_r, coeff_b, enabled=True)

    # Kiem chung: ap he so vao chinh anh trung binh, do lai profile.
    fixed = corrector.apply(np.clip(mean, 0, 255).astype(np.uint8))
    after = measure_ratio_profile(fixed.astype(np.float64), n_bins=args.bins)
    print('\n--- Sau khi sua (muc tieu: R/G va B/G phang, gan 1.00) ---')
    print('  %-12s %8s %8s' % ('r^2', 'R/G', 'B/G'))
    for p in after:
        print('  %.2f-%.2f    %8.3f %8.3f' % (p['r2_lo'], p['r2_hi'],
                                              p['rg'], p['bg']))

    rg_before = [p['rg'] for p in profile]
    rg_after = [p['rg'] for p in after]
    spread_before = max(rg_before) - min(rg_before)
    spread_after = max(rg_after) - min(rg_after)
    clipped = clipping_report(mean, corrector)
    print('\n  Bien do R/G tam->goc : %.3f  ->  %.3f  (giam %.0f%%)' % (
        spread_before, spread_after,
        100 * (1 - spread_after / max(spread_before, 1e-6))))
    print('  Pixel cham tran 255  : %.2f%%' % (100 * clipped))

    if args.preview:
        h, w = mean.shape[:2]
        before = np.clip(mean, 0, 255).astype(np.uint8)
        sep = np.full((h, 4, 3), 255, np.uint8)
        cv2.putText(before, 'TRUOC', (6, 20), 0, 0.6, (0, 0, 255), 2)
        shown = fixed.copy()
        cv2.putText(shown, 'SAU', (6, 20), 0, 0.6, (0, 255, 0), 2)
        cv2.imwrite(args.preview, np.hstack([before, sep, shown]))
        print('  Anh so sanh          : %s' % args.preview)

    if not sym['ok'] and not args.force:
        print('\nKHONG ghi file: kiem tra doi xung that bai. Dung --force neu')
        print('ban van muon ghi, nhung hay quay flat-field truoc.')
        return 2

    data = {
        'enabled': True,
        'coeff_r': coeff_r,
        'coeff_b': coeff_b,
        'max_gain': 3.0,
        'meta': {
            'mode': args.mode,
            'source': os.path.basename(args.source),
            'frames': n,
            'resolution': '%dx%d' % (mean.shape[1], mean.shape[0]),
            'rg_spread_before': round(spread_before, 4),
            'rg_spread_after': round(spread_after, 4),
            'symmetry_ok': bool(sym['ok']),
        },
    }
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    with io.open(args.out, 'w', encoding='utf-8') as fh:
        fh.write(yaml.safe_dump(data, default_flow_style=False,
                                allow_unicode=True))
    print('\nDa ghi: %s' % args.out)
    if args.mode == 'drive':
        print('LUU Y: he so tu --mode drive la TAM THOI. Quay flat-field roi')
        print('chay lai truoc khi chot tham so nguong mau.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
