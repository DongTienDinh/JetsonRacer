# -*- coding: utf-8 -*-
"""Ghi log CSV theo dung schema DB §7 (+ cot phuc vu Technical Paper).

Schema nay la HOP DONG: chot truoc khi bat dau thu so lieu va khong doi nua.
Doi cot giua chung ky dong nghia voi vut het so lieu da thu.

DB §9.1 cho BTC quyen doi log + ma nguon de xac minh -> log phai trung thuc:
ghi ca `fps` (vong chinh) lan `det_fps` (vong detect), khong lam dep so.
"""

import io
import os
import time

FIELDS = [
    'timestamp',
    't_rel',
    'frame_id',
    'fps',
    'det_fps',
    'latency_ms',
    'sign_latency_ms',
    'detected_object',
    'confidence',
    'decision',
    'control_output',
    'cte',
    # Them cho bam vach dut: du de danh gia lai ca luot chay ma khong can video.
    'cte_lookahead',   # lech tai diem ngam xa -> giai thich vi sao danh lai som
    'curvature',       # do cong -> giai thich vi sao bo ga
    'lane_found',      # 0/1
    'n_bands',         # so dai tim thay vach; 0-1 = sap mat vach
    'throttle',        # tach rieng khoi control_output de ve bieu do nhanh
    'drive_mode',      # THANG / CUA - giai thich vi sao ga cao hay thap
    'state',
    'event',
]


class RunLogger(object):
    def __init__(self, log_dir, task, run_name=None, flush_every=30):
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        if run_name is None:
            run_name = time.strftime('%Y%m%d_%H%M%S')
        self.path = os.path.join(log_dir, 'run_' + task + '_' + run_name + '.csv')
        self._fh = io.open(self.path, 'w', encoding='utf-8')
        self._fh.write(u','.join(FIELDS) + u'\n')
        self._flush_every = flush_every
        self._since_flush = 0
        self._t0 = None
        self.n_rows = 0

    def mark_start(self, t0=None):
        """Moc xuat phat hop le - `t_rel` tinh tu day."""
        self._t0 = time.time() if t0 is None else t0

    def write(self, **kwargs):
        now = kwargs.get('timestamp', time.time())
        t_rel = 0.0 if self._t0 is None else (now - self._t0)

        row = {
            'timestamp': '%.3f' % now,
            't_rel': '%.3f' % t_rel,
            'frame_id': kwargs.get('frame_id', 0),
            'fps': '%.2f' % float(kwargs.get('fps', 0.0)),
            'det_fps': '%.2f' % float(kwargs.get('det_fps', 0.0)),
            'latency_ms': '%.2f' % float(kwargs.get('latency_ms', 0.0)),
            'sign_latency_ms': (
                '' if kwargs.get('sign_latency_ms') is None
                else '%.2f' % float(kwargs['sign_latency_ms'])
            ),
            'detected_object': _clean(kwargs.get('detected_object', '')),
            'confidence': (
                '' if kwargs.get('confidence') is None
                else '%.3f' % float(kwargs['confidence'])
            ),
            'decision': _clean(kwargs.get('decision', '')),
            'control_output': _clean(kwargs.get('control_output', '')),
            'cte': '%.4f' % float(kwargs.get('cte', 0.0)),
            'cte_lookahead': (
                '' if kwargs.get('cte_lookahead') is None
                else '%.4f' % float(kwargs['cte_lookahead'])
            ),
            'curvature': (
                '' if kwargs.get('curvature') is None
                else '%.4f' % float(kwargs['curvature'])
            ),
            'lane_found': (
                '' if kwargs.get('lane_found') is None
                else ('1' if kwargs['lane_found'] else '0')
            ),
            'n_bands': (
                '' if kwargs.get('n_bands') is None
                else '%d' % int(kwargs['n_bands'])
            ),
            'throttle': (
                '' if kwargs.get('throttle') is None
                else '%.3f' % float(kwargs['throttle'])
            ),
            'drive_mode': _clean(kwargs.get('drive_mode', '')),
            'state': _clean(kwargs.get('state', '')),
            'event': _clean(kwargs.get('event', '')),
        }

        self._fh.write(u','.join([u'%s' % row[f] for f in FIELDS]) + u'\n')
        self.n_rows += 1
        self._since_flush += 1
        if self._since_flush >= self._flush_every:
            self._fh.flush()
            self._since_flush = 0

    def close(self):
        if self._fh and not self._fh.closed:
            self._fh.flush()
            self._fh.close()


def _clean(value):
    """CSV khong quote -> loai bo dau phay va xuong dong trong gia tri."""
    if value is None:
        return ''
    return str(value).replace(',', ';').replace('\n', ' ')
