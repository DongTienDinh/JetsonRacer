# -*- coding: utf-8 -*-
"""Launcher cho giao dien thu du lieu lai xe trong Jupyter.

Trong mot Jupyter cell dat tai thu muc goc du an, chay:

    %run tools/collect_drive_jupyter.py

De thu giao dien tren laptop ma KHONG dieu khien motor, dung mot cell Python:

    from tools.collect_drive_jupyter import launch
    collector = launch(source_kind='video', video_path='raw_camera.avi',
                       driver_kind='dryrun')
"""

from __future__ import print_function

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from jetracer_baseline.manual_collection import launch_manual_collection  # noqa: E402


def launch(config_path='configs/default.yaml', out_root='data/driving',
           source_kind='csi', video_path=None, driver_kind='nvidia',
           controller_index=0):
    return launch_manual_collection(
        config_path=config_path,
        out_root=out_root,
        source_kind=source_kind,
        video_path=video_path,
        driver_kind=driver_kind,
        controller_index=controller_index)


if __name__ == '__main__':
    try:
        get_ipython  # noqa: F821
    except NameError:
        print('Cong cu nay can chay trong Jupyter Notebook:')
        print('  %run tools/collect_drive_jupyter.py')
    else:
        collector = launch()
