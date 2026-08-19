# -*- coding: utf-8 -*-
"""Nhan dien doan thang / khuc cua roi chon ga va goc lai tuong ung.

VAN DE THUC TE tren sa ban: cac khuc cua RAT HEP. Bam lane bang PID thuan tuy
phan ung SAU khi da lech - den luc CTE du lon de PID danh het lai thi xe da cat
qua vach roi. Doan thang thi nguoc lai: PID giu xe giua vach nhung ga van bi
gioi han boi mot `v_max` chung, nen mat thoi gian o cho de an diem nhat.

GIAI PHAP - hai phan tach roi:

1. HAI MUC GA thay vi mot. `v_straight` cho doan thang, `v_corner` cho khuc cua,
   chuyen giua hai muc theo do cong CO TRE (hysteresis). Khong dung mot nguong
   duy nhat: do cong dao quanh nguong se lam ga bat/tat lien tuc, xe giat.

2. LAI FEED-FORWARD theo do cong, cong THANG vao dau ra PID:

       steer = PID(sai_so) + k_ff * do_cong

   PID chi sua cai da sai; feed-forward danh lai NGAY khi nhin thay duong cong,
   khong doi CTE tang. Tren cua hep day la khac biet giua bam duoc va cat cua.
   Trong che do CUA con nhan them `corner_steer_gain` de dat gan het lai.

Do cong den tu da thuc fit cua LaneDetector va da duoc gioi han trong vung thuc
su nhin thay vach, nen no BAO TRUOC khuc cua - ga giam truoc khi xe toi cua chu
khong phai phanh sau khi da lech.

Ga con duoc gioi han toc do doi: len tu tu, xuong nhanh. Bo ga phai duoc ngay
khi thay cua; len ga thi tu tu de banh khong truot.
"""

STRAIGHT = 'THANG'
CORNER = 'CUA'


class CornerController(object):
    def __init__(self, cfg):
        self.reset_from_config(cfg)
        self.mode = STRAIGHT
        self._throttle = 0.0
        self.steer_wanted = 0.0
        self.steer_limit = self.steer_max

    def reset_from_config(self, cfg):
        c = cfg.get
        # Hai muc ga. `v_straight` mac dinh lay `v_max` de config cu van chay.
        self.v_straight = float(c('control.v_straight', c('control.v_max', 0.20)))
        self.v_corner = float(c('control.v_corner', c('control.v_min', 0.10)))
        self.v_min = float(c('control.v_min', 0.10))
        self.v_max = float(c('control.v_max', 0.20))

        # Hai nguong khac nhau = hysteresis. enter > exit, neu khong se rung.
        self.curve_enter = float(c('control.curve_enter', 0.22))
        self.curve_exit = float(c('control.curve_exit', 0.14))
        if self.curve_exit >= self.curve_enter:
            # Cau hinh sai se lam mat hoan toan tac dung chong rung -> keo ve
            # mot khoang tre toi thieu thay vi im lang chay sai.
            self.curve_exit = self.curve_enter * 0.7

        self.slowdown = float(c('control.slowdown', 0.12))
        self.curve_feedforward = float(c('control.curve_feedforward', 0.9))
        self.corner_steer_gain = float(c('control.corner_steer_gain', 1.6))
        self.steer_max = float(c('control.steer_max', 0.60))
        # Tran lai RIENG cho khuc cua. Cua cua sa ban rat hep nen can be gan het
        # lai, nhung dung tran do cho ca doan thang thi xe giat va de vuot lane.
        # Mac dinh bang steer_max -> config cu khong doi hanh vi.
        self.corner_steer_max = float(
            c('control.corner_steer_max', self.steer_max))
        if self.corner_steer_max < self.steer_max:
            self.corner_steer_max = self.steer_max

        # Don vi: lenh ga / giay. Xuong nhanh hon len (xem docstring).
        self.throttle_rise_rate = float(c('control.throttle_rise_rate', 0.8))
        self.throttle_fall_rate = float(c('control.throttle_fall_rate', 3.0))

    def reset(self):
        self.mode = STRAIGHT
        self._throttle = 0.0
        self.steer_wanted = 0.0
        self.steer_limit = self.steer_max

    # ------------------------------------------------------------------ mode
    def update_mode(self, curvature):
        mag = abs(float(curvature))
        if self.mode == STRAIGHT:
            if mag >= self.curve_enter:
                self.mode = CORNER
        else:
            if mag <= self.curve_exit:
                self.mode = STRAIGHT
        return self.mode

    # ------------------------------------------------------------------- step
    def step(self, pid_output, cte, curvature, dt, lane_found=True):
        """Tra ve (steer, throttle, mode).

        `pid_output` la dau ra PID da tinh san - CornerController khong so huu
        PID, de vong goi quyet dinh dung sai so nao (cte hay tron voi diem ngam).
        """
        mode = self.update_mode(curvature)

        # --- lai: PID + feed-forward theo do cong -------------------------
        steer = pid_output + self.curve_feedforward * float(curvature)
        limit = self.steer_max
        if mode == CORNER:
            steer *= self.corner_steer_gain
            limit = self.corner_steer_max
        self.steer_wanted = steer          # truoc khi cat, de bao chan tran
        steer = max(-limit, min(limit, steer))
        self.steer_limit = limit

        # --- ga: chon muc theo che do, roi tru theo do lech ---------------
        target = self.v_corner if mode == CORNER else self.v_straight
        target -= self.slowdown * abs(float(cte))
        if not lane_found:
            # Mat vach thi khong duoc giu ga doan thang: xe dang lai theo gia
            # tri cu, cang nhanh cang xa vach.
            target = min(target, self.v_corner)
        target = max(self.v_min, min(self.v_max, target))

        # --- gioi han toc do doi ga ---------------------------------------
        if dt <= 0.0:
            dt = 1e-3
        delta = target - self._throttle
        if delta > 0:
            step = self.throttle_rise_rate * dt
            self._throttle += min(delta, step)
        else:
            step = self.throttle_fall_rate * dt
            self._throttle += max(delta, -step)
        self._throttle = max(0.0, min(self.v_max, self._throttle))

        return steer, self._throttle, mode

    @property
    def throttle(self):
        return self._throttle
