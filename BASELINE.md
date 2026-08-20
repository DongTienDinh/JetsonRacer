# BASELINE — Jetson AI Racer Challenge 2026

Tài liệu gốc: `Thể lệ.docx.pdf` (thể lệ khung) + `Đề bài chi tiết.docx.pdf` (tài liệu chuyên môn — **ưu tiên cao hơn** khi có mâu thuẫn, theo mục 7.3 thể lệ).

Đây là **baseline**: hệ thống đơn giản nhất chắc chắn ăn điểm, dùng làm mốc so sánh cho mọi cải tiến sau này (và là mục `baseline` bắt buộc trong Experimental Plan của paper).

---

## 1. Phân tích điểm — quyết định toàn bộ chiến lược kỹ thuật

### 1.1. Cơ cấu điểm tổng

| Hạng mục | Tỷ trọng | Ai chấm |
|---|---|---|
| Speed Track | 30% | Chạy thực tế |
| Smart City | **40%** | Chạy thực tế |
| Technical Paper | 20% | Hội đồng |
| Oral Defense | 10% | Hội đồng |

> **Kết luận 1:** Smart City nặng hơn Speed Track (40 vs 30) **và** là tiêu chí tie-break số 1 ở cả 2 tài liệu. Ưu tiên nguồn lực: Smart City > Speed Track.
>
> **Kết luận 2:** Paper + Defense = 30% — ngang Speed Track. Không phải phần phụ. Log phải được thiết kế từ ngày đầu để nuôi paper, không phải bịa lại vào phút chót.

### 1.2. Speed Track — quy đổi "giá" của từng lỗi

```
Điểm lượt = max(0, CP + FPS + Time − Penalty)
  CP    = số checkpoint hợp lệ × 10, tối đa 30
  FPS   = 10 nếu FPS trung bình pipeline ≥ 20, ngược lại 0   (nhị phân)
  Time  = 60 − 2 × (t/10)   → 0 điểm tại t = 300 s = đúng giới hạn 5 phút
```

Thời gian có giá **0.2 điểm/giây**. Quy đổi mọi lỗi ra điểm thực (điểm trừ + thời gian phạt):

| Vi phạm | Trừ điểm | Phạt giây | **Tổng thiệt hại thực** |
|---|---|---|---|
| Đụng chướng ngại vật | −5 | +10 s | **−7** |
| Lệch khỏi lane | −10 | +15 s | **−13** |
| Xuất phát trước hiệu lệnh | −10 | +10 s | **−12** |
| Đi ngược chiều / can thiệp tay | hủy lượt | — | **−100%** |

> **Kết luận 3 (quan trọng nhất):** 1 lần lệch lane = 13 điểm = **65 giây chạy chậm**. Chạy chậm hơn 1 phút vẫn lời hơn là lệch lane 1 lần. Baseline phải tối ưu *độ ổn định bám lane trước*, tốc độ sau. Đừng đua tốc độ ở lượt 1.

Mục tiêu điểm thực tế:

| Kịch bản | CP | FPS | Time | Penalty | Điểm | ×30% |
|---|---|---|---|---|---|---|
| Baseline an toàn (lap 120 s, 0 lỗi) | 30 | 10 | 36 | 0 | **76** | 22.8 |
| Tối ưu (lap 60 s, 0 lỗi) | 30 | 10 | 48 | 0 | **88** | 26.4 |
| Lap 60 s nhưng 2 lần lệch lane | 30 | 10 | 42 | 20 | **62** | 18.6 |

### 1.3. Smart City — thời gian gần như không đáng kể

```
Điểm lượt = max(0, Biển báo + Time − Penalty)
  Biển báo = (số biển đọc được / tổng số biển) × 70    [xem §6 - công thức gốc viết sai]
  Time     = 30 − 1 × (t/10), không âm  → 0 điểm tại t = 300 s
```

Thời gian có giá **0.1 điểm/giây**; biển báo chiếm 70/100.

- Chạy chậm hơn **100 giây** = −10 điểm = đúng bằng thiệt hại của việc **đọc sai 1 biển** (nếu có 7 biển).
- **Vượt đèn đỏ = hủy lượt = mất trắng.** Đây là rủi ro đơn lẻ lớn nhất của cả cuộc thi.
- Đi sai lộ trình = kết thúc lượt, chỉ tính điểm tới biển đúng xa vạch xuất phát nhất → sai 1 ngã tư là mất toàn bộ phần sau.

> **Kết luận 4:** Smart City **chạy chậm, đọc chắc**. Quy tắc an toàn của FSM: gặp đèn đỏ HOẶC không chắc chắn → DỪNG. Không bao giờ "đoán" khi tiến vào giao lộ.

### 1.4. Ngân sách thời gian ngày thi — không có slack

Tổng 25 phút: Speed Track ≤ 15 phút cho **3 lượt × 5 phút**, Smart City ≤ 10 phút cho **2 lượt × 5 phút**. Tức là **0 giây dự phòng** nếu lượt nào chạy hết giờ.

> **Kết luận 5:** Kịch bản chạy bắt buộc:
> - **Lượt 1 = lượt "gửi tiền"**: tham số bảo thủ nhất, mục tiêu duy nhất là *về đích không lỗi*. Có điểm rồi mới nói chuyện khác.
> - **Lượt 2–3**: tăng `v_max` theo profile đã test sẵn (config có sẵn, đổi bằng 1 flag, **không sửa code tại chỗ thi**).
> - Chuẩn bị sẵn script khởi động 1 lệnh; thời gian setup ăn vào thời gian thi.

---

## 2. Ràng buộc kỹ thuật cứng

| Ràng buộc | Giá trị | Nguồn | Ảnh hưởng thiết kế |
|---|---|---|---|
| Phần cứng | JetRacer AI Kit (Waveshare ROS AI Kit), **cấm sửa** | ĐB §2.1 | Không thêm cảm biến. Chỉ có 1 camera mono. |
| FPS pipeline | **≥ 20 FPS** (đọc ảnh + nhận diện + sinh lệnh) | ĐB §3.6 | Vòng điều khiển và vòng detect phải **tách luồng** |
| Latency biển báo → lệnh | **≤ 300 ms** | ĐB §4.9 | Detect ≥ 10 Hz là đủ; phải log mốc "biển xuất hiện rõ" |
| Điều khiển | Hoàn toàn tự động trong lượt thi | ĐB §2.2 | Không joystick/keyboard, kể cả để "sửa nhẹ" |
| Log | txt/csv, các trường quy định | ĐB §7 | Schema log là **hợp đồng** — xem §4 |
| Dataset BTC | **Không giống sa bàn thật** | ĐB §7 | Bắt buộc tự thu dữ liệu + augment mạnh |
| Xe dùng chung | 5 xe / 10 đội | Thể lệ §1.2 | **Offline-first**: phần lớn dev phải chạy được không cần xe |

**Lưu ý môi trường:** Jetson Nano trong JetRacer ROS AI Kit chạy **JetPack 4.5.1** (đã xác nhận trên xe ngày 2026-08-20: `tensorrt 7.1.3.0`, hostname `nano-4gb-jp451`) / Ubuntu 18.04 / **Python 3.6**. Toàn bộ code baseline trong repo này viết tương thích Python 3.6 (không dùng `dataclasses`, không dùng `X | Y`, không f-string `=`). Đây là bẫy rất hay gặp: code viết trên laptop Python 3.11 sẽ không chạy trên xe.

---

## 3. Kiến trúc baseline

```
                 ┌──────────────── CaptureThread (30 Hz) ────────────────┐
   CSI camera ──▶│ grab → giữ frame MỚI NHẤT, bỏ frame cũ (không queue)  │
                 └────────────────────────┬─────────────────────────────┘
                                          │ latest frame
        ┌─────────────────────────────────┼─────────────────────────────────┐
        │                                 │                                 │
 ┌──────▼───────┐                 ┌───────▼────────┐                ┌───────▼────────┐
 │ LANE (mỗi    │                 │ STOPLINE (mỗi  │                │ DETECT THREAD  │
 │ frame, CV cổ │                 │ frame, CV)     │                │ (10 Hz, YOLO)  │
 │ điển, ~3 ms) │                 │                │                │ → cache kết quả│
 └──────┬───────┘                 └───────┬────────┘                └───────┬────────┘
        │ cross-track error e             │ stopline_dist                   │ signs[]
        └─────────────────────────────────┴─────────────────┬───────────────┘
                                                            │
                                                  ┌─────────▼─────────┐
                                                  │  FSM quyết định   │
                                                  │  + latch biển báo │
                                                  └─────────┬─────────┘
                                                            │ target: steer, speed
                                                  ┌─────────▼─────────┐
                                                  │ PID → Driver      │
                                                  │ (servo + motor)   │
                                                  └─────────┬─────────┘
                                                            │
                                                  ┌─────────▼─────────┐
                                                  │ CSV Logger        │
                                                  └───────────────────┘
```

**Lý do tách luồng:** vòng điều khiển chính chạy ≥ 30 Hz (đảm bảo điểm FPS 10/10 và bám lane mượt), detector nặng chạy 10 Hz ở luồng riêng và chỉ *cache* kết quả. Vòng chính không bao giờ block chờ inference. 10 Hz detect → worst case 100 ms chờ + ~40 ms inference + 33 ms tick ≈ **175 ms < 300 ms** yêu cầu.

**Trung thực khi log:** log cả `fps` (vòng chính, đúng định nghĩa BTC: đọc ảnh + nhận diện + sinh lệnh) lẫn `det_fps` (vòng detect). Không "làm đẹp" số bằng cách bỏ bước nhận diện ra khỏi phép đo — BTC có quyền kiểm tra log và code.

### 3.1. Thuật toán baseline (cố tình đơn giản — phải chạy được)

| Khối | Baseline (B0) | Cải tiến ứng viên (để so sánh trong paper) |
|---|---|---|
| Bám lane | Bird's-eye warp → threshold → histogram peak → cross-track error | CNN regression (resnet18 road_following), IPM + polyfit bậc 2 |
| Điều khiển | PID trên `e`, tốc độ giảm theo `\|e\|` và độ cong | Pure Pursuit, Stanley, MPC nhẹ |
| Biển báo | YOLOv8n 320×320 TensorRT FP16 | YOLOv8s, SSD-MobileNet, + fallback màu/hình (xanh tròn = biển lệnh, đỏ tròn gạch = biển cấm) |
| Đèn giao thông | HSV mask vùng đỏ/xanh trong ROI đèn + xác nhận k/n frame | Đưa vào YOLO như 2 class |
| Vạch dừng | Dải trắng ngang trong ROI dưới | Học cùng detector |
| Vật cản | Class trong YOLO + bias điểm ngắm sang bên trống | Depth ước lượng mono, occupancy grid |
| Quyết định giao lộ | FSM + latch biển bằng vote k-of-n, khóa khi bbox đủ lớn | Behavior tree, rule engine từ map |

**Vì sao lane dùng CV cổ điển ở baseline, không dùng CNN:** dataset BTC "không giống sa bàn thật" (ĐB §7). CV cổ điển chỉnh được tại chỗ trong 5 phút bằng 2–3 tham số ngưỡng; CNN cần thu data + train lại — không kịp trong ngày thi. CNN là *cải tiến*, không phải baseline.

---

## 4. Schema log (hợp đồng — không đổi sau khi bắt đầu thu số liệu)

Đúng các trường ĐB §7 yêu cầu, cộng thêm trường phục vụ paper. Một dòng CSV mỗi frame vòng chính:

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `timestamp` | float | epoch giây, độ phân giải ms |
| `t_rel` | float | giây kể từ lúc xuất phát hợp lệ |
| `frame_id` | int | số thứ tự frame |
| `fps` | float | FPS trung bình trượt của vòng chính |
| `det_fps` | float | FPS của luồng detect |
| `latency_ms` | float | thời gian xử lý frame này (đọc ảnh → sinh lệnh) |
| `sign_latency_ms` | float | **biển xuất hiện rõ → lệnh tương ứng** (chỉ điền tại frame ra quyết định) |
| `detected_object` | str | nhãn: `lane`, `stopline`, tên biển, `obstacle` |
| `confidence` | float | độ tin cậy model |
| `decision` | str | `straight`/`left`/`right`/`stop`/`avoid`/`follow` |
| `control_output` | str | `steer=<-1..1>;throttle=<0..1>` |
| `cte` | float | cross-track error chuẩn hoá (−1..1) — **metric chính cho paper** |
| `state` | str | state hiện tại của FSM |
| `event` | str | `start`,`checkpoint_1..3`,`lane_departure`,`restart`,`red_stop`,`finish` |

`tools/analyze_log.py` đọc file này và xuất thẳng bảng + biểu đồ cho paper.

---

## 5. Kế hoạch triển khai (tuần tương đối tới ngày thi T-0)

Gán ngày thi thật vào T-0 rồi lùi ngược. Nếu quỹ thời gian ngắn hơn 6 tuần, cắt từ P4 trở lên, **không bao giờ cắt P0/P1**.

| Phase | Tuần | Mục tiêu ra được | Definition of Done |
|---|---|---|---|
| **P0 — Dựng nền** | T-6 | Repo chạy được ở chế độ `replay` trên laptop; schema log chốt; xác nhận API điều khiển thật của xe | `python -m jetracer_baseline.cli replay --video x.mp4` chạy ra CSV hợp lệ |
| **P1 — Bám lane** | T-5 | Xe chạy hết 1 vòng Speed Track, tốc độ thấp, 0 lệch lane | 5/5 lần thử về đích không lệch lane; FPS ≥ 25 |
| **P2 — Dữ liệu + biển báo** | T-4 | Tự thu ≥ 1500 ảnh có nhãn trên sa bàn thật; YOLOv8n mAP@0.5 ≥ 0.85 | Model chạy TensorRT trên xe ≥ 15 Hz, latency p95 ≤ 200 ms |
| **P3 — Smart City FSM** | T-3 | Hoàn thành lộ trình đô thị, dừng đúng đèn đỏ | 5/5 lần dừng đúng đèn đỏ; 0 lần vượt đèn đỏ |
| **P4 — Tối ưu tốc độ** | T-2 | Profile `fast` cho Speed Track | Lap time giảm ≥ 30% mà không phát sinh lệch lane |
| **P5 — Tổng duyệt + Paper** | T-1 | Diễn tập đúng format ngày thi; paper bản nháp đầy đủ | Chạy trọn 25 phút theo kịch bản thi; paper đủ 8 mục IEEE |
| **T-0** | Ngày thi | — | Chỉ đổi config, **không sửa code** |

### 5.1. Phân công (đội 3–5 người)

| Vai trò | Sở hữu | Không đụng vào |
|---|---|---|
| Perception-Lane | `perception/lane.py`, `stopline.py`, tuning ngưỡng | model biển báo |
| Perception-Signs | dataset, train YOLO, `perception/signs.py`, export TensorRT | control |
| Control & FSM | `control/`, `fsm.py`, profile tốc độ | perception nội bộ |
| Data & Paper | `tools/analyze_log.py`, thí nghiệm, biểu đồ, viết paper | — |
| Team lead | tích hợp, kịch bản ngày thi, làm việc với BTC | — |

Người viết paper **phải** là người chạy `analyze_log.py` — tránh cảnh cuối kỳ không ai hiểu số liệu (§8.2 thể lệ chấm rất nặng phần "phân tích định lượng qua log").

### 5.2. Rủi ro & phương án dự phòng

| Rủi ro | Xác suất | Ảnh hưởng | Dự phòng |
|---|---|---|---|
| Ánh sáng sa bàn thi khác lúc train | Cao | Mất lane / mất biển | Auto-exposure **lock** + chuẩn hoá kênh V; 3 preset ngưỡng chọn bằng flag; 5 phút chuẩn bị trên sa bàn dùng để chọn preset |
| Chỉ 5 xe / 10 đội → thiếu giờ chạy thật | Cao | Không kịp tune | Offline-first: replay video + unit test; mỗi buổi có xe phải có **kế hoạch thí nghiệm viết sẵn**, không mò tại chỗ |
| Vượt đèn đỏ | Trung bình | **Hủy lượt** | FSM fail-safe: đỏ hoặc không chắc → dừng; ngưỡng confidence cho `green` cao hơn `red` |
| FPS < 20 khi bật detector | Trung bình | −10 điểm | Tách luồng (đã thiết kế); TensorRT FP16; input 320; watchdog hạ tần số detect khi FPS tụt |
| Dataset BTC lệch domain | Chắc chắn | mAP thấp | Tự thu trên sa bàn thật + augment mạnh (brightness/blur/perspective); fallback màu-hình |
| Python 3.6 trên Jetson | Chắc chắn | Code không chạy | Ràng buộc cú pháp từ đầu + CI check |
| Xe hỏng ngày thi | Thấp | Mất lượt | ĐB §6 cho phép BTC xử lý; báo trọng tài ngay, ghi biên bản |

---

## 6. Điểm cần hỏi BTC (gửi trước ngày thi)

Đây là những chỗ **thật sự mơ hồ hoặc mâu thuẫn trong tài liệu**, không phải hỏi cho có:

1. **Mâu thuẫn công thức Speed Track.** §3.6 ghi `Điểm Checkpoint = số CP × 10, tối đa 30`, nhưng ví dụ §8 tính `3 checkpoint → 90 điểm`. 10 hay 30 điểm/checkpoint? (Không đổi chiến thuật — vẫn phải ăn đủ 3 — nhưng đổi mô hình dự báo điểm.)
2. **Công thức biển báo Smart City sai đơn vị.** §4.7 ghi `số biển đọc được × (tổng số biển / 70)`. Với N biển, đọc đủ N chỉ ra 70 điểm khi N = 70. Ví dụ §8 lại cho 70 điểm khi về đích. Có phải ý là `× (70 / tổng số biển)` không?
3. **"Biển báo đọc được" định nghĩa thế nào?** Nhận diện đúng nhãn là đủ, hay phải *hành động đúng* theo biển? Biển ở nhánh đường xe không đi qua có tính không?
4. **Thời gian phạt có cộng vào `t` khi tính Điểm thời gian không?** §2.4 định nghĩa "Thời gian hoàn thành" đã gồm phạt — xác nhận là dùng luôn `t` đó cho công thức điểm.
5. **Điểm thời gian rời rạc hay liên tục?** `mỗi 10 giây trừ 2 điểm` là `floor(t/10)` hay `t/10`?
6. **FPS đo bằng gì?** Log của đội hay thiết bị BTC? Cửa sổ tính trung bình là toàn lượt hay trượt?
7. **Đèn giao thông:** chu kỳ cố định hay điều khiển tay? Đỏ kéo dài bao lâu? Có đèn vàng không?
8. **Smart City có cho biết trước bản đồ/lộ trình không**, hay lộ trình hoàn toàn suy ra từ biển báo tại chỗ? ("Đi sai lộ trình" ngụ ý có lộ trình đúng xác định trước.)
9. **Speed Track chạy 1 vòng hay nhiều vòng?** §3.5 ghi "một vòng" — xác nhận.
10. **Chướng ngại vật:** kích thước, màu, số lượng, có cố định vị trí giữa các lượt không?

---

## 7. Cấu trúc repo & cách chạy

```
BASELINE.md                      ← file này
docs/
  de-bai-rut-gon.md              ← checklist yêu cầu rút từ 2 PDF
  ke-hoach-thuc-nghiem.md        ← thí nghiệm + metric + ánh xạ sang paper
configs/
  default.yaml                   ← profile "safe" (dùng cho lượt 1)
  fast.yaml                      ← profile tốc độ (lượt 2–3)
src/jetracer_baseline/
  config.py  camera.py  pipeline.py  fsm.py  logging_csv.py  cli.py
  perception/  lane.py  stopline.py  signs.py
  control/     pid.py   driver.py
tools/
  collect_dataset.py             ← thu ảnh có timestamp để gán nhãn
  analyze_log.py                 ← CSV → bảng + biểu đồ cho paper
tests/test_smoke.py
```

Chạy thử không cần xe (Windows/laptop):

```bash
python -m src.jetracer_baseline.cli replay --source synthetic --task speed --frames 200 --config configs/default.yaml
```

Chạy trên xe:

```bash
python3 -m src.jetracer_baseline.cli run --task smartcity --config configs/default.yaml --driver nvidia
```

Phân tích log sau mỗi lượt:

```bash
python tools/analyze_log.py logs/run_*.csv --out reports/
```

---

## 8. Việc cần làm ngay (theo thứ tự)

1. Gán ngày thi thật vào T-0 trong §5, chốt lịch lùi.
2. Gửi 10 câu hỏi §6 cho BTC.
3. **Ngày đầu tiên có xe:** xác nhận API điều khiển thật (`nvidia_racecar` hay ROS `/cmd_vel` hay lib Waveshare) và Python version → cập nhật `control/driver.py`. Mọi thứ khác phụ thuộc vào việc này.
4. Quay ≥ 3 video toàn vòng sa bàn ở 3 điều kiện sáng khác nhau → làm bộ replay để dev offline.
5. Chốt schema log §4 và không đổi nữa.
