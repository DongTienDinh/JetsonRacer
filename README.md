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
# Lần đầu chỉ test servo, không cộng thêm tải motor
python3 tools/check_hardware.py --driver nvidia --actuator \
  --wheels-are-lifted --steering-only
```

Preflight sẽ từ chối test actuator nếu Jetson không ở power mode 5W. Thiết lập
bằng `sudo nvpmodel -m 1`, rồi xác nhận lại với `nvpmodel -q`.

Tune bám vạch **trực tiếp trên JupyterLab** — xem camera xe đang chạy, kéo slider,
thấy ngay kết quả, không cần SSH gõ lệnh trên Jetson:

```bash
jupyter lab --ip=0.0.0.0 --no-browser
```

Mở [tune_lane.ipynb](tune_lane.ipynb) từ máy khác và Run All. Giao diện mở lên ở trạng
thái **dừng** — không lệnh nào xuống phần cứng cho đến khi bấm **CHẠY - BÁM LINE**; ga
tăng dần trong 1 giây đầu. Chỉnh xong bấm **LƯU CONFIG** → ghi `configs/tuned.yaml`,
rồi chạy lượt chính thức có ghi log. Trong giao diện, mỗi lần bấm **CHẠY** hoặc
**LÁI TAY** sẽ tự tạo một CSV riêng; **DỪNG**, dừng khẩn cấp, lỗi camera/model/driver
và đóng UI đều ghi dòng kết rồi flush/đóng file:

```bash
python3 -m src.jetracer_baseline.cli run --task speed --driver nvidia --override configs/tuned.yaml --record
```

`FPS(XỬ LÝ)` trong giao diện đếm các `frame_id` mới đã hoàn tất camera → perception
→ sinh lệnh. Panel/JPEG chạy worker latest-only 5 Hz và không nằm trong phép đo.
Nếu AUTO không có completion mới trong 0,25 giây, xe tự cắt motor và chốt log với
`event=camera_stall`; chỉnh ngưỡng bằng `tuning.pipeline_stall_s` khi thật sự cần.
Log tune dùng để tìm bottleneck; số nghiệm thu chính thức vẫn lấy từ một lượt CLI
trên Jetson thật (không dùng replay không realtime).

Đo an toàn trên CSI thật nhưng không quay bánh, chạy 3 lượt 60 giây:

```bash
python3 -m src.jetracer_baseline.cli run --task speed --driver dryrun \
  --override configs/cnn.yaml --max-seconds 60
python3 tools/analyze_log.py "logs/run_speed_*.csv"
```

Mục tiêu có dự phòng: `FPS mean >= 24`, `FPS p05 >= 20`, latency `p95 <= 25 ms`.
Mốc chấm là 20 FPS, nhưng cấu hình chỉ vừa 20 sẽ dễ tụt dưới ngưỡng khi nóng máy
hoặc đồng thời ghi video.

Hiệu chuẩn lens shading màu (**làm trước khi tune bất kỳ ngưỡng màu nào**) —
camera CSI của xe đỏ gấp ~1.6 lần ở góc ảnh so với tâm ảnh, nên một ngưỡng HSV
duy nhất không dùng được cho cả khung hình nếu chưa sửa:

```bash
python3 tools/collect_dataset.py --mode video --source csi --session flatfield --out data/calib --seconds 15
```

```bash
python tools/calib_shading.py --mode flatfield --source data/calib/flatfield_<ts>.avi --preview reports/shading.png
```

Quy trình đầy đủ kèm tiêu chí đạt bằng số: [docs/quy-trinh-thu-data-sa-ban.md](docs/quy-trinh-thu-data-sa-ban.md).

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

```bash
jupyter notebook --ip=0.0.0.0 --port=8889 --no-browser
```

Sau đó mở `collect_drive.ipynb` từ giao diện Jupyter và chạy lần lượt từng cell.
Notebook đã có sẵn kiểm tra đường dẫn, launcher và cell in trạng thái chẩn đoán.
Trước khi dựng giao diện, notebook chạy `tools/check_hardware.py` để mở camera
3 giây, ghi `reports/camera_sample.jpg`, sau đó khởi tạo driver thật và ghi
neutral (`steering=0`, `throttle=0`). Nếu bước này thất bại thì không ARM xe.

Kiểm tra widget trước khi chạy:

```bash
python3 -c "import ipywidgets, traitlets; print(ipywidgets.__version__)"
```

Nếu thiếu, cài trong venv đã tạo bằng `--system-site-packages` để không thay thế
OpenCV/CUDA/TensorRT của JetPack:

```bash
pip install "ipywidgets>=7.5,<8" "traitlets>=4.3"
```

Giao diện đi theo thứ tự `KIỂM TRA TAY CẦM → kê xe và TEST SERVO/MOTOR →
MỞ CAMERA → đặt xe xuống đất → ARM TAY CẦM → BẮT ĐẦU GHI`.
Checkbox `BÁNH XE ĐÃ KÊ KHỎI MẶT ĐẤT` chỉ khóa hai nút test actuator, không
khóa ARM lái thật. Dead-man mặc định tắt. Ga tay mặc định bắt đầu ở `0.12` sau
deadzone và tăng tới `0.30`; hiệu chuẩn `Ga khởi động` tới mức nhỏ nhất làm bánh
vừa quay khi xe đang kê. Profile low-power dùng expo `0.45`, giới hạn lệnh `0.60`
và slew-rate `1.5 đơn vị/s`. Quan trọng hơn, driver chặn cứng output servo ở
`[-0.40, +0.40]`; với PWM `750–2250 µs`, xung thực tế không vượt khoảng
`1200–1800 µs` dù lệnh đến từ UI, PID hay FSM — đây là mức an toàn tạm thời lúc
bring-up, không phải giới hạn cơ khí thật của servo.

Mỗi slider lái/ga (`Deadzone`, `Lái tối đa`, `Độ mềm lái`, `Ga khởi động`,
`Ga tối đa`, `Test lái`, `Test ga`) đi kèm một ô số bên cạnh, gõ số chính xác
thay vì chỉ kéo chuột; ô số và slider dùng chung khoảng giá trị nên không kẹt
ở mức mặc định bảo thủ của `configs/default.yaml`. Bên dưới mục *Test phần
cứng độc lập* có panel **NÂNG CAO** để chỉnh trực tiếp `steering_gain`,
`steering_offset`, `steering_output_min/max` và `throttle_gain` — đây mới là
lớp quyết định góc lái/tốc độ thật (các slider phía trên chỉ là hệ số nhân
*trước* lớp này), bấm **ÁP DỤNG GIỚI HẠN NÂNG CAO** để ghi thẳng vào driver
đang chạy, không cần mở lại kernel. Thay đổi chỉ có hiệu lực trong phiên hiện
tại (không ghi vào YAML) và tự DISARM xe sau khi áp dụng để bắt buộc ARM lại
có chủ đích. **Trước khi tin số mới lúc lái thật, luôn kiểm tra bằng TEST
SERVO (bánh đã kê) hoặc `tools/check_hardware.py --calibrate-steering`** —
nới `steering_output_max` mà chưa xác nhận cơ khí là đúng cơ chế đã gây
reboot do sụt áp khi servo kẹt chân (xem mục *Bẻ lái rồi Jetson/Jupyter “tắt”*
bên dưới). Mỗi session được lưu riêng tại
`data/driving/<session_timestamp>/` gồm
ảnh gốc, `labels.csv`, `drive.avi`, `drive.sidecar.csv` và `metadata.json`.
Video và ảnh JPEG đều được ghi ở thread riêng (mỗi loại một hàng đợi giới hạn)
nên không chặn vòng camera/điều khiển; nếu thẻ SD/eMMC ghi chậm, hệ thống thà
bỏ vài mẫu (đếm ở `samples_dropped` / `video_frames_dropped` trong
`metadata.json`) còn hơn làm khựng cam hoặc trễ lệnh lái. Sidecar nối từng
frame video với `frame_id`, thời gian và lệnh lái/ga. Mặc định video ghi 15 FPS
để giảm tải Jetson, còn `Lưu FPS` chỉ điều khiển tần suất ảnh JPEG/nhãn. Với dataset segmentation bài 1, để
`Lưu FPS = 5`; nếu sau này train imitation learning thì tăng lên 15–20 FPS.

Hai bảng `Axes live` và `Buttons đang bấm` luôn cập nhật trước cả khi ARM. Nếu
`TEST SERVO/MOTOR` không làm phần cứng chuyển động với backend `nvidia`, lỗi nằm
ở driver/nguồn/dây xe chứ không nằm ở mapping tay cầm. Các nút test bị khóa cho
đến khi người vận hành xác nhận `BÁNH XE ĐÃ KÊ KHỎI MẶT ĐẤT`.

Camera ghi rõ backend thực tế trên giao diện: `csi-gstreamer` hoặc
`usb-v4l2-index-0`. Trường hợp OpenCV báo camera đã open nhưng frame đầu thất
bại, code release và mở lại Argus trước khi fallback; exception trong thread
camera được đưa ra log và có thể bấm `MỞ CAMERA` để thử lại. Camera capture ở
mode `1280x720@30`, rồi `nvvidconv` resize về `640x480`; không yêu cầu cảm biến
CSI phát trực tiếp mode xử lý. Project dùng khóa `/tmp/jetracer_camera.lock` để
ngăn hai tiến trình của chính project tranh camera.

Nếu log có `Failed to create CaptureSession`, đóng **mọi kernel** từng dùng
camera (chỉ đóng tab trình duyệt là chưa đủ), rồi chạy:

```bash
sudo systemctl restart nvargus-daemon
sleep 2
python3 tools/check_hardware.py --driver nvidia --camera-seconds 3
```

Backend mặc định không còn phụ thuộc class `NvidiaRacecar` đã cài trong image;
nó dùng thẳng `adafruit_servokit` vốn có sẵn trên image JetRacer. Adapter thư
viện cũ chỉ còn là đường dự phòng khi đổi `control.driver.implementation`.

Backend `nvidia` của project điều khiển trực tiếp PCA9685 duy nhất ở `0x40`,
channel 0 cho lái và channel 1 cho ga; không còn tạo object motor `0x60` từ
biến thể thư viện cũ. Driver có watchdog: mất lệnh UI quá 0.8 giây thì tự ghi
ga về 0. Tần số PWM để trống nhằm giữ mặc định 50 Hz của ServoKit; chỉ ép 60 Hz
sau khi đã đo/kiểm chứng đúng servo và ESC. Tắt kernel Jupyter cũ rồi chạy lại
notebook sau mỗi lần cập nhật code.
Nếu không kết nối được driver, kiểm tra phần cứng trước:

```bash
sudo i2cdetect -y -r 1
```

Kết quả phải có `40`. Nếu không có `40`, đây là lỗi nguồn/cáp/tiếp xúc I²C,
không phải lỗi tay cầm hay camera.

Nếu bẻ lái làm mất giao diện hoặc xe tắt, mở một Terminal riêng trước khi thử:

```bash
python3 tools/diagnose_shutdown.py monitor
```

Sau sự cố hoặc sau khi dừng monitor, chạy `python3 tools/diagnose_shutdown.py check`.
`REBOOT CONFIRMED` nghĩa là boot ID đã đổi: ưu tiên xử lý pin, rail 5 V, servo
kẹt/chạm chặn cơ khí và power mode; đó không phải exception của notebook.

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
| Driver phần cứng | **Đã chốt: PCA9685 `0x40` trực tiếp qua ServoKit**, channel lái/ga `0/1`, có emergency lock + watchdog |

Hai việc chặn tiến độ, theo thứ tự: **(1)** quay video sa bàn để dev offline, **(2)** thu dataset biển báo.
