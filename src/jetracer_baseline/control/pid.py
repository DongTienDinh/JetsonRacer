# -*- coding: utf-8 -*-
"""PID mot chieu cho goc lai.

Baseline dat Ki = 0 co chu dich: thanh phan tich phan de bi windup o cua gap
va lam xe dao khi vao lane lai sau khi mat lane. Chi bat Ki khi thi nghiem E2
chung minh duoc loi ich - va khi do phai co chong windup.
"""


class PID(object):
    def __init__(self, kp, ki=0.0, kd=0.0, out_limit=1.0, integral_limit=0.5):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.out_limit = float(out_limit)
        self.integral_limit = float(integral_limit)
        self._integral = 0.0
        self._prev_error = None

    def reset(self):
        self._integral = 0.0
        self._prev_error = None

    def step(self, error, dt):
        if dt <= 0.0:
            dt = 1e-3

        p = self.kp * error

        self._integral += error * dt
        if self._integral > self.integral_limit:
            self._integral = self.integral_limit
        elif self._integral < -self.integral_limit:
            self._integral = -self.integral_limit
        i = self.ki * self._integral

        if self._prev_error is None:
            d = 0.0
        else:
            d = self.kd * (error - self._prev_error) / dt
        self._prev_error = error

        out = p + i + d
        if out > self.out_limit:
            out = self.out_limit
        elif out < -self.out_limit:
            out = -self.out_limit
        return out
