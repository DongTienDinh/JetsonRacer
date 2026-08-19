# -*- coding: utf-8 -*-
"""Launcher cho giao dien tune bam vach trong Jupyter/JupyterLab.

Tren XE (Jetson), trong mot cell tai thu muc goc du an:

    %run tools/tune_lane_jupyter.py

Mac dinh la `driver_kind='dryrun'`: xem camera va chinh nguong thoai mai, banh
KHONG quay. Chi khi nao muon cho xe chay that moi truyen `driver_kind='nvidia'`:

    from tools.tune_lane_jupyter import launch
    ui = launch(driver_kind='nvidia')

Tren LAPTOP, tune lai bang video da quay o sa ban (khong can xe):

    from tools.tune_lane_jupyter import launch
    ui = launch(source_kind='video', video_path='raw_camera.avi')

Chinh xong bam LUU CONFIG -> ghi configs/tuned.yaml, roi chay that bang:

    python3 -m src.jetracer_baseline.cli run --task speed --driver nvidia \\
        --override configs/tuned.yaml
"""

from __future__ import print_function

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from jetracer_baseline.tuning_ui import launch_tuning_ui  # noqa: E402


def launch(config_path='configs/default.yaml', source_kind='csi',
           video_path=None, driver_kind='dryrun',
           save_path='configs/tuned.yaml'):
    """Mo giao dien. `driver_kind='dryrun'` mac dinh de khong lo cho xe chay."""
    return launch_tuning_ui(config_path=config_path, source_kind=source_kind,
                            video_path=video_path, driver_kind=driver_kind,
                            save_path=save_path)


if __name__ == '__main__':
    try:
        get_ipython  # noqa: F821
    except NameError:
        print('Cong cu nay can chay trong Jupyter Notebook/JupyterLab:')
        print('  %run tools/tune_lane_jupyter.py')
        print()
        print('De kiem tra phan xu ly ma khong can Jupyter, dung engine truc tiep:')
        print('  python -c "import sys; sys.path.insert(0, \'src\'); '
              'from jetracer_baseline.tuning_ui import LaneTuningEngine; '
              'print(LaneTuningEngine().proc_size)"')
    else:
        ui = launch()
