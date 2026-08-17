# JetRacer Baseline — Jetson AI Racer Challenge 2026

Baseline chạy được cho hai bài thi **Speed Track** và **Smart City**.

**Đọc [BASELINE.md](BASELINE.md) trước** — phân tích điểm ở §1 quyết định toàn bộ các lựa chọn kỹ thuật trong repo này (tóm tắt: *độ ổn định quan trọng hơn tốc độ rất nhiều*).

| Tài liệu | Nội dung |
|---|---|
| [BASELINE.md](BASELINE.md) | Phân tích điểm, chiến lược, kiến trúc, kế hoạch, rủi ro, câu hỏi cho BTC |
| [docs/quy-trinh-test-xe.md](docs/quy-trinh-test-xe.md) | **Bring-up xe từng bước**, an toàn → rủi ro |
| [docs/de-bai-rut-gon.md](docs/de-bai-rut-gon.md) | Checklist yêu cầu rút từ 2 PDF gốc |
| [docs/ke-hoach-thuc-nghiem.md](docs/ke-hoach-thuc-nghiem.md) | Thí nghiệm E1–E6, metric, ánh xạ sang Technical Paper |

---

## Cài đặt

**Trên laptop** (dev + phân tích log):

```bash
pip install -r requirements.txt
```

**Trên Jetson Nano** — OpenCV/numpy đã có sẵn trong JetPack, **đừng** pip install đè (sẽ phá bản OpenCV có CUDA/GStreamer của NVIDIA):

```bash
pip3 install PyYAML
```

---

## Chạy thử ngay, không cần xe

```bash
python -m src.jetracer_baseline.cli replay --source synthetic --frames 200
```

Chạy trên video đã quay ở sa bàn (xử lý đủ mọi frame, kết quả tất định):

```bash
python -m src.jetracer_baseline.cli replay --source video --video data/lap1.mp4
```

Chạy test tích hợp:

```bash
python tests/test_smoke.py
```

---

## Chạy trên xe

Quy trình đầy đủ ở **[docs/quy-trinh-test-xe.md](docs/quy-trinh-test-xe.md)** — làm theo đúng thứ tự, đừng nhảy bước.

Bring-up tự động (môi trường → config → camera → driver):

```bash
python3 tools/check_hardware.py --driver nvidia
```

Test servo và động cơ — **kê xe lên giá, bánh không chạm đất**:

```bash
python3 tools/check_hardware.py --driver nvidia --actuator --wheels-are-lifted
```

Tune ngưỡng bám lane trên sa bàn thật (xuất ảnh 4 ô từng bước xử lý):

```bash
python3 tools/tune_lane.py --source csi --frames 5 --out reports/lane
```

Rồi chạy với `--driver nvidia` (backend đã chốt theo `control.txt` của BTC — xem [BASELINE.md](BASELINE.md)):

```bash
python3 -m src.jetracer_baseline.cli run --task speed --driver nvidia
```

Ngày thi, lượt 2–3 Speed Track (chỉ đổi config, **không sửa code**):

```bash
python3 -m src.jetracer_baseline.cli run --task speed --driver nvidia --override configs/fast.yaml
```

Trước mỗi lần đồng bộ code lên xe — chặn cú pháp Python 3.7+ lọt vào (Jetson chạy Python 3.6):

```bash
python tools/check_py36.py
```

---

## Sau mỗi lượt chạy

```bash
python tools/analyze_log.py "logs/run_speed_*.csv" --out reports/ --lane-departures 0
```

In ra CTE rms/p95, FPS mean/p05, latency p50/p95, sign latency p95, và **điểm mô phỏng theo đúng công thức BTC**. Vẽ biểu đồ timeline dùng thẳng trong paper.

Thu dữ liệu để train biển báo:

```bash
python3 tools/collect_dataset.py --session chieu_nang --out data/raw --every 0.2
```

Thu dữ liệu lái xe đồng bộ `ảnh + steering + throttle` trong Jupyter:

```python
%run tools/collect_drive_jupyter.py
```

Kiểm tra widget trước khi chạy:

```bash
python3 -c "import ipywidgets, traitlets; print(ipywidgets.__version__)"
```

Nếu thiếu, cài trong venv đã tạo bằng `--system-site-packages` để không thay thế
OpenCV/CUDA/TensorRT của JetPack:

```bash
pip install "ipywidgets>=7.5,<8" "traitlets>=4.3"
```

Giao diện bắt buộc đi theo thứ tự `MỞ CAMERA → ARM TAY CẦM → BẮT ĐẦU GHI`.
Mặc định phải giữ nút `LB` (`button 4`) thì xe mới nhận ga và ga bị giới hạn ở
`0.20`. Mỗi session được lưu riêng tại `data/driving/<session_timestamp>/` gồm
ảnh gốc, `labels.csv` và `metadata.json`. Với dataset segmentation bài 1, để
`Lưu FPS = 5`; nếu sau này train imitation learning thì tăng lên 15–20 FPS.

Chạy thử giao diện bằng video trên laptop, hoàn toàn không điều khiển motor:

```python
from tools.collect_drive_jupyter import launch
collector = launch(source_kind='video', video_path='raw_camera.avi',
                   driver_kind='dryrun')
```

---

## Trạng thái baseline

| Thành phần | Trạng thái |
|---|---|
| Bám lane (CV cổ điển) | Chạy được, **chưa tune trên sa bàn thật** |
| PID + profile tốc độ | Chạy được, `v_max` là giá trị tạm — chốt bằng thí nghiệm E3 |
| FSM Smart City | Chạy được, có test cho đèn đỏ/xanh và biển lệnh |
| Nhận diện biển báo | **Chỉ có backend `stub`** — cần dataset + train YOLO (phase P2) |
| Log + phân tích | Hoàn chỉnh, đúng schema ĐB §7 |
| Driver phần cứng | **Đã chốt: `nvidia`** (`NvidiaRacecar`, `steering_gain=-1.0`) — theo `control.txt` của BTC. Vẫn nên chạy `probe` ngày đầu có xe để xác nhận thư viện có sẵn |

Hai việc chặn tiến độ, theo thứ tự: **(1)** quay video sa bàn để dev offline, **(2)** thu dataset biển báo.
