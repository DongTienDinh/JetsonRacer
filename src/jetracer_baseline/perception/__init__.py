# -*- coding: utf-8 -*-
"""Cac khoi nhan thuc. Xem `build_lane_detector` de chon nguon bam vach."""

from __future__ import print_function


def build_lane_detector(cfg):
    """Chon nguon mask bam vach theo `lane.mode`.

    `color_center` / `gray` -> LaneDetector (CV co dien, chay o dau cung).
    `cnn`                   -> CnnLaneDetector (TensorRT, CHI tren xe).

    Import LUOI o trong nhanh, khong o dau file: `lane_cnn` keo theo tensorrt va
    pycuda, hai thu chi co tren Jetson. Import o dau file se lam moi test tren
    laptop gay ngay khi khoi dong, ke ca khi khong dinh dung CNN.

    Moi noi tao detector deu phai goi ham nay - neu co cho tu goi thang
    LaneDetector() thi doi `lane.mode: cnn` se im lang khong co tac dung o cho
    do, va ban se tuong minh dang do CNN trong khi thuc ra van la CV.
    """
    from .lane import LaneDetector
    if str(cfg.get('lane.mode', 'color_center')) != 'cnn':
        return LaneDetector(cfg)
    from .lane_cnn import CnnLaneDetector
    return CnnLaneDetector(cfg)
