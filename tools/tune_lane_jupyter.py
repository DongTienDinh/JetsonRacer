# -*- coding: utf-8 -*-
"""Launcher cho giao dien tune bam vach trong Jupyter/JupyterLab.

Tren XE (Jetson), trong mot cell tai thu muc goc du an:

    %run tools/tune_lane_jupyter.py

Mo len la o trang thai DUNG - khong co lenh nao xuong phan cung. Bam nut
**2. CHAY - BAM LINE** thi xe moi bat dau tu chay. Ga tang dan trong 1 giay
dau de xe khong giat.

Muon xem thu ma chac chan banh khong quay (vi du khi dat xe tren ban):

    from tools.tune_lane_jupyter import launch
    ui = launch(driver_kind='dryrun')

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
           video_path=None, driver_kind=None,
           save_path='configs/tuned.yaml', soft_start_s=1.0,
           controller_index=0, data_root='data/driving'):
    """Mo giao dien tune.

    KHONG co lenh nao xuong phan cung cho den khi bam nut CHAY. `driver_kind`
    de trong -> camera that dung 'nvidia', replay video dung 'dryrun'.
    """
    return launch_tuning_ui(config_path=config_path, source_kind=source_kind,
                            video_path=video_path, driver_kind=driver_kind,
                            save_path=save_path, soft_start_s=soft_start_s,
                            controller_index=controller_index,
                            data_root=data_root)


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
