# Kế hoạch thực nghiệm & metric

Mục đích kép: (1) tune hệ thống có căn cứ thay vì mò, (2) sinh ra đúng số liệu mà Technical Paper (20%) và Oral Defense (10%) cần.

Nguyên tắc: **mỗi buổi có xe đều phải có kế hoạch thí nghiệm viết trước.** Xe dùng chung 5 chiếc / 10 đội — thời gian trên xe là tài nguyên khan hiếm nhất của cả dự án, không được dùng để mò tham số.

---

## 1. Metric

### 1.1. Perception

| Metric | Định nghĩa | Đo ở đâu | Ngưỡng đạt |
|---|---|---|---|
| `CTE_rms` | RMS của cross-track error chuẩn hoá (−1..1) trên toàn lượt | cột `cte` trong log | ≤ 0.15 |
| `CTE_p95` | Phân vị 95 của \|cte\| | log | ≤ 0.35 |
| `lane_loss_rate` | % frame không tìm được lane | log (`detected_object` rỗng) | ≤ 2% |
| `mAP@0.5` | Trên tập test tự thu ở sa bàn thật | offline eval | ≥ 0.85 |
| `sign_acc_onboard` | Độ chính xác nhãn *khi chạy trên xe* (đối chiếu video) | log + video | ≥ 0.95 |
| `red_recall` | Recall của đèn đỏ (bỏ sót = hủy lượt) | offline eval | **= 1.00** |
| `green_precision` | Precision của đèn xanh (báo nhầm xanh = vượt đèn đỏ) | offline eval | ≥ 0.99 |

> `red_recall` và `green_precision` là **hai metric an toàn**, không đánh đổi với bất kỳ thứ gì. Vượt đèn đỏ = hủy lượt = mất 40% trọng số.

### 1.2. Hệ thống

| Metric | Định nghĩa | Ngưỡng đạt |
|---|---|---|
| `fps_mean` | FPS trung bình vòng chính (đọc ảnh + nhận diện + sinh lệnh) | **≥ 20** (điểm nhị phân) |
| `fps_p05` | Phân vị 5 — đo độ ổn định, không chỉ trung bình | ≥ 18 |
| `latency_p50/p95` | Thời gian xử lý 1 frame | p95 ≤ 45 ms |
| `sign_latency_p95` | Biển hiện rõ → lệnh tương ứng | **≤ 300 ms** |
| `cpu/gpu/temp` | Tài nguyên Jetson (`tegrastats`) | không throttle |

### 1.3. Nhiệm vụ

| Metric | Định nghĩa | Ngưỡng đạt |
|---|---|---|
| `lap_time` | Thời gian hoàn thành vòng Speed Track | xem mục 3 |
| `departures_per_lap` | Số lần lệch lane / vòng | **0** |
| `obstacle_hits` | Số lần chạm vật cản | 0 |
| `intersection_acc` | % giao lộ rẽ đúng hướng | 100% |
| `success_rate` | % lượt hoàn thành hợp lệ trên tổng số lượt thử | ≥ 90% |
| `score_sim` | Điểm mô phỏng theo công thức BTC (`tools/analyze_log.py`) | — |

---

## 2. Bộ thí nghiệm

Mỗi thí nghiệm: **giả thuyết → biến độc lập → biến kiểm soát → số lần lặp → metric quyết định**. Tối thiểu **5 lần lặp** cho mọi so sánh; báo cáo mean ± std, không báo cáo lần chạy tốt nhất.

### E1 — Baseline bám lane (CV cổ điển) vs CNN regression

| | |
|---|---|
| Giả thuyết | CV cổ điển đạt `CTE_rms` tương đương CNN nhưng ổn định hơn khi đổi điều kiện sáng, và tune nhanh hơn tại chỗ |
| Biến độc lập | phương pháp ∈ {bird's-eye + histogram, resnet18 regression} |
| Kiểm soát | cùng video/lượt, cùng PID, cùng tốc độ |
| Lặp | 5 lượt × 2 phương pháp |
| Quyết định | `CTE_rms`, `lane_loss_rate`, `fps_mean`, độ suy giảm khi đổi sáng |

→ Bảng này vào thẳng mục **Results** của paper. Đây chính là phần "so sánh mô hình, phân tích trade-off" mà Thể lệ §8.2 chấm.

### E2 — Độ nhạy của PID

| | |
|---|---|
| Giả thuyết | Tồn tại vùng `(Kp, Kd)` cho `CTE_rms` thấp mà không dao động ở tốc độ cao |
| Biến độc lập | `Kp` ∈ {0.4, 0.6, 0.8, 1.0}, `Kd` ∈ {0.05, 0.1, 0.2} |
| Kiểm soát | cùng lane pipeline, cùng `v_max` |
| Lặp | 3 lượt/cấu hình (grid 12 điểm — chạy trên **replay** trước, chỉ đem 3 cấu hình tốt nhất lên xe) |
| Quyết định | `CTE_rms`, biên độ dao động steering (std của `steer`) |

→ Heatmap `Kp × Kd` là biểu đồ dễ ăn điểm nhất trong paper. Chạy phần lớn offline để tiết kiệm giờ xe.

### E3 — Đánh đổi tốc độ ↔ độ ổn định (trực tiếp ra chiến thuật ngày thi)

| | |
|---|---|
| Giả thuyết | Tồn tại `v_max` mà điểm mô phỏng đạt cực đại; vượt ngưỡng đó `departures` tăng nhanh hơn phần điểm thời gian kiếm được |
| Biến độc lập | `v_max` ∈ {0.15, 0.20, 0.25, 0.30, 0.35} |
| Lặp | 5 lượt/mức |
| Quyết định | `score_sim` (dùng đúng công thức BTC, 0.2 điểm/giây, −13/lần lệch lane) |

→ Đây là thí nghiệm **quan trọng nhất về mặt điểm số**. Kết quả chốt luôn giá trị `v_max` cho `configs/default.yaml` (lượt 1) và `configs/fast.yaml` (lượt 2–3).

### E4 — Detector: kích thước model × input size

| | |
|---|---|
| Biến độc lập | {YOLOv8n, YOLOv8s} × {256, 320, 416} × {FP16 TensorRT, FP32 PyTorch} |
| Quyết định | `mAP@0.5`, `fps`, `sign_latency_p95` trên Jetson thật |

→ Bảng Pareto accuracy-vs-latency. Chọn điểm nhỏ nhất thoả `sign_latency_p95 ≤ 300 ms` **và** `red_recall = 1.0`.

### E5 — Robust với domain shift (ánh sáng)

| | |
|---|---|
| Giả thuyết | Augment mạnh + auto-exposure lock giảm suy giảm hiệu năng khi đổi điều kiện sáng |
| Biến độc lập | {không augment, augment cơ bản, augment mạnh} × {sáng ngày, đèn phòng, ngược sáng} |
| Quyết định | độ tụt `mAP` và `lane_loss_rate` giữa các điều kiện |

→ Trực tiếp trả lời rủi ro lớn nhất: ĐB §7 nói rõ dataset BTC **không giống sa bàn thật**. Thí nghiệm này là phần **Discussion/Limitation** của paper.

### E6 — Ablation FSM an toàn

| | |
|---|---|
| Biến độc lập | {không vote, vote 3-of-5, vote 5-of-7} × {có/không ngưỡng bbox-area} |
| Quyết định | `intersection_acc`, số lần quyết định sai, số lần dừng thừa |

---

## 3. Mục tiêu hiệu năng theo phase

| Phase | `lap_time` | `departures/lap` | `fps_mean` | `mAP@0.5` | `score_sim` ST |
|---|---|---|---|---|---|
| P1 | ≤ 180 s | 0 | ≥ 25 | — | ~64 |
| P2 | ≤ 150 s | 0 | ≥ 22 | ≥ 0.80 | ~70 |
| P3 | ≤ 120 s | 0 | ≥ 22 | ≥ 0.85 | ~76 |
| P4 | ≤ 80 s | ≤ 0.2 | ≥ 20 | ≥ 0.85 | ~84 |
| P5 | ≤ 70 s | 0 | ≥ 20 | ≥ 0.90 | ~86 |

---

## 4. Dataset tự thu

ĐB §7: dataset BTC **sẽ không giống sa bàn thi**. Kế hoạch tự thu là bắt buộc, không phải tuỳ chọn.

| Hạng mục | Mục tiêu |
|---|---|
| Số ảnh biển báo | ≥ 1500 ảnh, ≥ 150 ảnh/class |
| Class | `turn_left`, `turn_right`, `go_straight`, `no_left`, `no_right`, `no_straight`, `red_light`, `green_light`, `obstacle` |
| Điều kiện | ≥ 3 mức sáng × ≥ 3 khoảng cách × ≥ 3 góc lệch |
| Chia tập | train/val/test = 70/15/15, **tách theo buổi thu** (không random split — tránh rò rỉ frame gần giống nhau) |
| Video lane | ≥ 3 video toàn vòng/điều kiện sáng, dùng làm bộ replay offline |
| Công cụ | `tools/collect_dataset.py` (thu, đặt tên theo timestamp) + LabelImg/Roboflow |

> Tách tập **theo buổi thu**, không random. Random split trên frame video liên tiếp làm mAP val cao giả tạo — và đây đúng là loại lỗi mà giám khảo sẽ hỏi ở Oral Defense.

---

## 5. Ánh xạ sang Technical Paper

| Mục paper | Nguồn số liệu |
|---|---|
| Introduction | Phân tích điểm ở `BASELINE.md` §1 — nêu rõ vì sao độ ổn định > tốc độ |
| Related Work | Lane following cổ điển vs học sâu; YOLO/SSD trên edge; PID/Pure Pursuit/Stanley |
| Method | Kiến trúc `BASELINE.md` §3 + sơ đồ tách luồng |
| Experimental Setup | Mục 1–2 tài liệu này (metric, biến, số lần lặp, cách chia tập) |
| Results | E1 (bảng), E2 (heatmap), E3 (đường cong score-vs-speed), E4 (Pareto) |
| Discussion | E5 — domain shift; phân tích ca lỗi từ log |
| Limitation | Camera mono, không đo khoảng cách; FSM luật cứng không tổng quát; dữ liệu 1 sa bàn |
| References | Chỉ tài liệu có thật, kiểm chứng được (Thể lệ §9.1) |

**Quy tắc bắt buộc:** mọi con số trong paper phải truy ngược được về một file log cụ thể trong `logs/`. Không có số nào được gõ tay. Thể lệ §9.2 cấm "số liệu không có thật", và §9.1 cho BTC quyền đòi log + mã nguồn để xác minh.
