# Quy trình test xe

Thứ tự từ **an toàn → rủi ro**. Mỗi bước chỉ làm khi bước trước đã qua. Đừng nhảy thẳng xuống bước 6 — lỗi ở bước 1–3 sẽ biểu hiện thành "xe chạy loạn" và bạn sẽ debug nhầm chỗ.

> **Xe dùng chung 5 chiếc cho 10 đội.** Giờ trên xe là tài nguyên khan hiếm nhất của cả dự án. Bước 0 làm sạch ở nhà; bước 1–7 mới cần xe.

---

## Bước 0 — Ở nhà, không cần xe

```bash
python tests/test_smoke.py
python tools/check_py36.py
python -m src.jetracer_baseline.cli replay --source synthetic --frames 200
```

Cả ba phải sạch trước khi đụng vào xe. `check_py36.py` là bước dễ quên nhất: Jetson Nano chạy **Python 3.6**, code viết trên laptop 3.10 chạy ngon ở nhà rồi chết ngay dòng `import` trên xe — đúng lúc đang tính giờ dùng chung.

---

## Bước 1 — Kết nối Jetson

```bash
ssh jetson@<ip-cua-xe>
```

Ghi lại ngay các thông tin này (cần cho mục *Experimental Setup* của paper):

```bash
python3 --version
cat /etc/nv_tegra_release
python3 -c "import cv2; print(cv2.__version__)"
free -h && nvpmodel -q
```

Chuyển code lên xe:

```bash
rsync -av --exclude='__pycache__' --exclude='logs' --exclude='data' \
  ./ jetson@<ip>:~/jetracer-baseline/
```

Trên xe cài đúng **một** thứ — OpenCV/numpy đã có sẵn trong JetPack, pip install đè sẽ phá bản có CUDA/GStreamer của NVIDIA:

```bash
pip3 install PyYAML
```

---

## Bước 2 — Bring-up tự động

```bash
python3 tools/check_hardware.py --skip-camera
```

Chạy qua: môi trường Python → config → backend điều khiển. Backend **đã chốt là `nvidia`** (theo `control.txt` của BTC) — bước này chỉ để xác nhận thư viện `jetracer` có sẵn trên xe, không phải để chọn lại backend.

Nếu `nvidia` báo `KHONG`, tìm xem xe dùng thư viện gì:

```bash
python3 -c "import jetracer; print(jetracer.__file__)"
ls /opt/ros/*/setup.bash 2>/dev/null && rostopic list | grep -i cmd
pip3 list 2>/dev/null | grep -iE "jetracer|jetbot|rosmaster|waveshare"
```

Tìm được rồi thì cập nhật [src/jetracer_baseline/control/driver.py](../src/jetracer_baseline/control/driver.py) — file đã để sẵn 3 adapter, chỉ cần sửa/thêm đúng một class.

---

## Bước 3 — Camera

```bash
python3 tools/check_hardware.py --driver nvidia --camera-seconds 5
```

**Mở `reports/camera_sample.jpg` ra xem bằng mắt.** Ba câu hỏi:

- Ảnh có bị lộn ngược / lật gương không? → sửa `flip_method` trong [camera.py](../src/jetracer_baseline/camera.py)
- Có thấy mặt đường **ngay trước mũi xe** không? Nếu camera ngóc quá cao, `roi_top` sẽ vô dụng
- FPS camera có ≥ 25 không? Vòng điều khiển cần ≥ 20 FPS để ăn 10 điểm (ĐB §3.6) — camera chậm là nút thắt đầu tiên, không phải thuật toán

---

## Bước 4 — Servo và động cơ ⚠

> **KÊ XE LÊN GIÁ. BÁNH XE KHÔNG ĐƯỢC CHẠM ĐẤT.** Bước này làm bánh quay.

```bash
python3 tools/check_hardware.py --driver nvidia --actuator --wheels-are-lifted
```

Script quét góc lái giữa → trái → giữa → phải → giữa (động cơ không chạy), rồi một xung ga 0.6 giây. **Bạn phải tự kiểm tra bằng mắt:**

| Quan sát | Nếu sai thì sửa |
|---|---|
| Bánh đánh **trái** khi in `TRAI` | Đảo dấu `steering` trong `driver.py` |
| `steering = 0` thì bánh thẳng | Chỉnh `control.driver.steering_offset` trong `configs/default.yaml` |
| Bánh sau quay **tiến** | Đảo dấu `throttle` |

Hiệu chuẩn `steering_offset` ở đây, không phải lúc đang chạy. Servo lệch 0.05 là đủ để xe trôi dần ra mép lane sau nửa vòng — mà lệch lane tốn **13 điểm**.

---

## Bước 5 — Tune lane (vẫn kê xe, chưa cho chạy)

Đặt xe lên sa bàn thật, ở đúng tư thế bám lane:

```bash
python3 tools/tune_lane.py --source csi --frames 5 --out reports/lane
```

Xuất ảnh ghép 4 ô mỗi frame. **Mở ra xem, đừng chỉ nhìn tỉ lệ phần trăm:**

| Triệu chứng | Chỉnh |
|---|---|
| Ô `binary` toàn trắng hoặc toàn đen | `lane.threshold_c` |
| Ô `ROI` không thấy mặt đường | `lane.roi_top` |
| Ô `birds-eye` hai mép đường không gần song song | `lane.warp_src` |
| `cte` không đổi dấu khi đẩy xe lệch trái/phải | Warp hoặc ROI sai — **dừng lại sửa**, đừng chạy tiếp |

Quét nhanh nhiều ngưỡng (chú ý dấu `=`, vì giá trị âm):

```bash
python3 tools/tune_lane.py --source csi --sweep-c=-18,-12,-6 --frames 3
```

Kiểm tra thủ công quan trọng nhất: **cầm xe đẩy sang trái, `cte` phải âm; đẩy sang phải, `cte` phải dương.** Sai dấu ở đây thì PID sẽ lái xe ra khỏi lane nhanh hơn là vào.

Trước khi rời sa bàn, quay **≥ 3 video toàn vòng ở 3 điều kiện sáng khác nhau**:

```bash
python3 tools/collect_dataset.py --mode video --session sang_som --seconds 120 --out data/video
```

Để vừa lái tay vừa lưu ảnh và nhãn điều khiển đồng bộ, mở notebook tại thư mục
gốc dự án rồi chạy:

```bash
jupyter notebook --ip=0.0.0.0 --port=8889 --no-browser
```

Mở `collect_drive.ipynb` trong giao diện vừa khởi động và chạy từng cell từ trên
xuống dưới.

Notebook bắt buộc chạy camera/driver check riêng trước khi tạo collector. Kết
quả đúng phải có `Camera OK`, backend (`csi-gstreamer` hoặc
`usb-v4l2-index-0`), FPS, ảnh `reports/camera_sample.jpg` và thông báo driver đã
khởi tạo/ghi neutral. Nếu bước này lỗi thì không bỏ qua cell để ARM xe. Đóng mọi
kernel/notebook camera cũ; chỉ khi camera đã được release mới thử
`sudo systemctl restart nvargus-daemon`.

Riêng lỗi `gstnvarguscamerasrc ... Failed to create CaptureSession` là lỗi
Argus chưa tạo được phiên camera, không phải lỗi tay cầm. Làm đúng thứ tự:

```bash
# Trong Jupyter: Kernel > Shut Down Kernel cho moi notebook da dung camera
sudo systemctl restart nvargus-daemon
sleep 2
cd /home/jetson/JetsonRacer
python3 tools/check_hardware.py --driver nvidia --camera-seconds 3
```

Không chạy đồng thời lệnh trên và nút `MỞ CAMERA`. Mã mới dùng capture mode
`1280x720@30`, retry Argus một lần và khóa liên tiến trình để tránh hai collector
của project cùng chiếm camera. Backend mặc định dùng trực tiếp ServoKit trên
PCA9685 `0x40`; class `NvidiaRacecar` của image không còn nằm trên đường chạy
mặc định vì biến thể hai địa chỉ có thể xung đột trên board một PCA9685.

Thứ tự trên giao diện: `KIỂM TRA TAY CẦM` → kê xe và tick xác nhận →
`TEST SERVO` → `TEST MOTOR 0.5S` → `MỞ CAMERA` → đặt xe xuống đất →
`ARM TAY CẦM` → `BẮT ĐẦU GHI`. Dead-man mặc định tắt; checkbox bánh đã kê chỉ
là gate của hai nút test, không can thiệp ARM lái thật. `Ga khởi động` mặc định
0.12 và `Ga tối đa` 0.30; hiệu chuẩn ngưỡng khởi động khi xe vẫn đang kê.
Hai bảng `Axes live` và `Buttons đang bấm` dùng để tìm mapping
thật của từng loại tay cầm; không giả định mọi tay cầm đều là axes 2/1 và button
4. Nút `DỪNG KHẨN CẤP` được tuần tự hóa với control loop: hủy bài test, dừng
ghi, ghi ga=0 rồi DISARM; control loop không thể ghi đè lệnh ga sau emergency.
Watchdog cũng tự cắt ga nếu mất lệnh quá 0.8 giây. Dữ liệu nằm trong
`data/driving/<session_timestamp>/`; không
chia ngẫu nhiên frame của cùng session sang cả train và validation.

Đây là thứ cho phép cả đội làm việc offline suốt tuần sau mà không cần tranh xe.

---

## Bước 6 — Chạy thật, tốc độ thấp ⚠

> Người cầm sẵn tay để chộp xe. Bắt đầu bằng `v_max` rất thấp.

```bash
python3 -m src.jetracer_baseline.cli run --task speed --driver nvidia --max-seconds 30
```

Lần đầu nên hạ tốc hơn nữa:

```bash
python3 -m src.jetracer_baseline.cli run --task speed --driver nvidia \
  --max-seconds 30 --log-dir logs/first
```

(sửa `control.v_max` xuống `0.12` trong config trước khi chạy)

Sau mỗi lượt, **luôn** phân tích log:

```bash
python3 tools/analyze_log.py "logs/first/*.csv" --out reports/
```

Ba con số cần nhìn:

| Chỉ số | Ngưỡng | Nếu không đạt |
|---|---|---|
| `FPS mean` | ≥ 20 | Giảm `proc_width`, hạ `detect_hz` |
| `CTE rms` | ≤ 0.15 | Tune PID (thí nghiệm E2) hoặc lane (bước 5) |
| `mat lane` | ≤ 2% | Quay lại bước 5 |

---

## Bước 7 — Tăng tốc dần

Chỉ tăng `v_max` khi đã **5/5 lượt về đích không lệch lane**. Chạy thí nghiệm E3 trong [ke-hoach-thuc-nghiem.md](ke-hoach-thuc-nghiem.md): quét `v_max ∈ {0.15, 0.20, 0.25, 0.30, 0.35}`, 5 lượt mỗi mức, chọn theo `score_sim` chứ không theo lap time.

Lý do phải chọn theo điểm mô phỏng: nhanh hơn 10 giây được **2 điểm**, nhưng lệch lane 1 lần mất **13 điểm**. Mắt thường nhìn xe chạy nhanh thấy "tốt hơn" trong khi điểm thực tế thấp đi.

---

## Checklist trước khi rời phòng lab

- [ ] Log của mọi lượt đã copy về máy cá nhân (thư mục `logs/` không nằm trong git)
- [ ] Đã ghi lại `v_max`, `threshold_c`, `steering_offset` đang dùng — và **giá trị nào ứng với xe số mấy**
- [ ] Đã quay video làm bộ replay offline
- [ ] Đã ghi vào nhật ký thí nghiệm: hôm nay thay đổi gì, kết quả ra sao

> Xe dùng chung 5 chiếc: `steering_offset` của xe #1 gần như chắc chắn khác xe #3. Ngày thi nhận xe nào thì phải hiệu chuẩn lại bước 4 trong 5 phút chuẩn bị. Ghi sẵn bảng trim theo từng xe.

---

## Sự cố hay gặp

| Triệu chứng | Nguyên nhân thường gặp |
|---|---|
| `No module named 'jetracer.nvidia_racecar'` | Repo NVIDIA chưa nằm trong Python path; kiểm tra `~/jetracer/jetracer/nvidia_racecar.py` hoặc đặt `JETRACER_NVIDIA_ROOT` |
| `No I2C device at address: 0x60` | Đang chạy code/kernel cũ. Backend mới chỉ dùng PCA9685 `0x40`; pull code, Shut Down toàn bộ kernel rồi chạy lại. Không thấy `40` trong `sudo i2cdetect -y -r 1` thì kiểm tra nguồn/cáp I²C |
| `Failed to create CaptureSession` / camera 0 frame | Đóng mọi kernel camera, restart `nvargus-daemon`; nếu vẫn lỗi thì tắt nguồn và kiểm tra chiều cáp CSI |
| FPS tụt dần theo thời gian | Jetson bị throttle nhiệt — kiểm tra `tegrastats`, bật quạt |
| Xe rẽ ngược hướng | Sai dấu `steering` (bước 4) hoặc sai dấu `cte` (bước 5) |
| Xe trôi dần ra mép dù đường thẳng | `steering_offset` chưa hiệu chuẩn |
| Xe giật lắc quanh tim đường | `Kp` quá cao hoặc `Kd` quá thấp — thí nghiệm E2 |
| Báo `unstuck` liên tục trong log | `fsm.motion_threshold` chưa hiệu chuẩn cho xe này |
