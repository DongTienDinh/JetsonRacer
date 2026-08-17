# -*- coding: utf-8 -*-
"""May trang thai quyet dinh cho ca hai bai thi.

Nguyen tac an toan (suy ra truc tiep tu bang loi DB §3.7 va §4.8):
  1. Khong chac chan  -> DUNG. Dung thua ton ~0.1 diem/giay o Smart City;
     vuot den do ton CA LUOT.
  2. Uu tien khong lech lane hon la chay nhanh: 1 lan lech = 13 diem = 65 giay.
  3. Tu thoat ket TRUOC moc 10 giay (DB §6), vi dong ho khong dung khi restart.
"""

import time

# --- Speed Track ---
S_IDLE = 'IDLE'
S_LANE = 'LANE_FOLLOW'
S_AVOID = 'AVOID'
S_UNSTUCK = 'UNSTUCK'
S_FINISH = 'FINISHED'
# --- Smart City ---
S_APPROACH = 'APPROACH_STOPLINE'
S_WAIT_RED = 'WAIT_RED'
S_TURN = 'TURNING'

MANDATORY = {
    'turn_left': 'left',
    'turn_right': 'right',
    'go_straight': 'straight',
}
PROHIBITORY = ('no_left', 'no_right', 'no_straight')


class Command(object):
    def __init__(self, steer_override=None, throttle_scale=1.0, decision='follow',
                 state=S_LANE, event='', force_stop=False):
        self.steer_override = steer_override   # None = de PID bam lane quyet dinh
        self.throttle_scale = throttle_scale   # nhan vao v_max
        self.decision = decision               # ghi vao cot `decision` cua log
        self.state = state
        self.event = event
        self.force_stop = force_stop


class DecisionFSM(object):
    def __init__(self, cfg, task='speed'):
        self.task = task
        self.state = S_IDLE

        self.turn_duration = float(cfg.get('fsm.turn_duration_s', 1.6))
        self.turn_steer = float(cfg.get('fsm.turn_steer', 0.75))
        self.stop_hold = float(cfg.get('fsm.stop_hold_s', 0.3))
        self.stuck_timeout = float(cfg.get('fsm.stuck_timeout_s', 3.0))
        self.unstuck_duration = float(cfg.get('fsm.unstuck_duration_s', 0.8))
        self.stopline_trigger = float(cfg.get('fsm.stopline_trigger', 0.35))
        # Bien cam -> huong thay the. DB §4.3: tai nga tu do "chi co hai huong".
        # Chua biet bo tri sa ban that -> de config, chot lai sau buoi khao sat.
        self.prohibit_fallback = cfg.get('fsm.prohibit_fallback', {
            'no_left': 'straight',
            'no_right': 'straight',
            'no_straight': 'right',
        })

        self._t_state = time.time()
        self._pending_turn = None
        self._t_no_motion = None
        self._last_event = ''

    # ------------------------------------------------------------------ helpers
    def _to(self, state, now):
        if state != self.state:
            self.state = state
            self._t_state = now

    def _elapsed(self, now):
        return now - self._t_state

    def _turn_command(self, direction, event=''):
        """Lenh lai cho mot huong re.

        Dung chung cho ca frame BAT DAU re lan cac frame duy tri, de frame dau
        tien cua cu re khong bi tre mot nhip - do dung la luc quan trong nhat.
        """
        if direction == 'left':
            steer = -self.turn_steer
        elif direction == 'right':
            steer = self.turn_steer
        else:
            steer = None       # di thang: van de PID bam lane dan duong
        return Command(steer_override=steer, throttle_scale=0.7,
                       decision=direction, state=S_TURN, event=event)

    def start(self, now=None):
        """Goi khi co hieu lenh xuat phat cua BTC (DB §3.3)."""
        now = time.time() if now is None else now
        self._to(S_LANE, now)
        return 'start'

    def finish(self, now=None):
        now = time.time() if now is None else now
        self._to(S_FINISH, now)
        return 'finish'

    # --------------------------------------------------------------------- step
    def step(self, lane, stopline=None, sign=None, moving=True, now=None):
        """
        lane     : LaneResult
        stopline : StoplineResult hoac None
        sign     : (label, confidence, t_first_seen) hoac None
        moving   : xe co dang thuc su di chuyen khong (uoc luong tu chenh lech anh)
        """
        now = time.time() if now is None else now

        if self.state == S_IDLE:
            return Command(steer_override=0.0, throttle_scale=0.0,
                           decision='stop', state=S_IDLE, force_stop=True)

        if self.state == S_FINISH:
            return Command(steer_override=0.0, throttle_scale=0.0,
                           decision='stop', state=S_FINISH, force_stop=True)

        # --- Watchdog ket xe: tu thoat truoc moc 10 s cua DB §6 -----------------
        if self.state != S_WAIT_RED:
            if not moving:
                if self._t_no_motion is None:
                    self._t_no_motion = now
                elif (now - self._t_no_motion) >= self.stuck_timeout:
                    self._t_no_motion = None
                    self._to(S_UNSTUCK, now)
            else:
                self._t_no_motion = None

        if self.state == S_UNSTUCK:
            if self._elapsed(now) >= self.unstuck_duration:
                self._to(S_LANE, now)
            else:
                # Lui nhe + danh lai nguoc huong lech de tu go
                steer = -self.turn_steer if lane.cte >= 0 else self.turn_steer
                return Command(steer_override=steer, throttle_scale=-1.0,
                               decision='unstuck', state=S_UNSTUCK, event='unstuck')

        # --- Dang thuc thi cu re ----------------------------------------------
        if self.state == S_TURN:
            if self._elapsed(now) >= self.turn_duration:
                self._pending_turn = None
                self._to(S_LANE, now)
            else:
                return self._turn_command(self._pending_turn or 'straight')

        # --- Dung den do -------------------------------------------------------
        if self.state == S_WAIT_RED:
            label = sign[0] if sign else None
            if label == 'green_light' and self._elapsed(now) >= self.stop_hold:
                self._to(S_LANE, now)
                return Command(throttle_scale=0.6, decision='straight',
                               state=S_LANE, event='green_go')
            # Mac dinh: van dung. Khong thay den => van dung (nguyen tac an toan 1).
            return Command(steer_override=0.0, throttle_scale=0.0, decision='stop',
                           state=S_WAIT_RED, force_stop=True)

        # --- Smart City: xu ly tin hieu ---------------------------------------
        if self.task == 'smartcity' and sign is not None:
            label = sign[0]
            if label == 'red_light':
                near = stopline is not None and stopline.found \
                    and stopline.distance <= self.stopline_trigger
                if near or self.state == S_APPROACH:
                    self._to(S_WAIT_RED, now)
                    return Command(steer_override=0.0, throttle_scale=0.0,
                                   decision='stop', state=S_WAIT_RED,
                                   event='red_stop', force_stop=True)
                # Thay do tu xa: bo ga, tien cham toi vach
                self._to(S_APPROACH, now)
                return Command(throttle_scale=0.35, decision='stop',
                               state=S_APPROACH, event='red_seen')

            if label in MANDATORY:
                self._pending_turn = MANDATORY[label]
            elif label in PROHIBITORY:
                self._pending_turn = self.prohibit_fallback.get(label, 'straight')

            if self._pending_turn is not None:
                near = stopline is None or not stopline.found \
                    or stopline.distance <= self.stopline_trigger
                if near:
                    self._to(S_TURN, now)
                    return self._turn_command(self._pending_turn,
                                              event='turn_' + self._pending_turn)
                return Command(throttle_scale=0.6, decision='follow',
                               state=S_APPROACH, event='sign_' + label)

        # --- Speed Track: tranh vat can ---------------------------------------
        if self.task == 'speed' and sign is not None and sign[0] == 'obstacle':
            # Lech diem ngam sang phia trong hon; giam toc de kip xu ly.
            bias = 0.45 if lane.cte <= 0 else -0.45
            self._to(S_AVOID, now)
            return Command(steer_override=lane.cte + bias, throttle_scale=0.55,
                           decision='avoid', state=S_AVOID, event='obstacle')

        if self.state in (S_AVOID, S_APPROACH) and self._elapsed(now) > 1.0:
            self._to(S_LANE, now)

        # --- Bam lane binh thuong ---------------------------------------------
        if not lane.found:
            # Mat lane: bo ga, giu huong cu. Dung han neu keo dai (an toan 2).
            scale = 0.35 if self._elapsed(now) < 1.5 else 0.0
            return Command(throttle_scale=scale, decision='follow',
                           state=self.state, event='lane_lost')

        self._to(S_LANE, now)
        return Command(throttle_scale=1.0, decision='follow', state=S_LANE)
