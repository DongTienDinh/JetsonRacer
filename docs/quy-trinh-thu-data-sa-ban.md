# Quy trình thu data trên sa bàn thật — Bài 1 Speed Track

Tài liệu này nối tiếp `docs/quy-trinh-test-xe.md`. Nó trả lời đúng một câu hỏi:
**cần cho xe chạy những gì trên sa bàn để biết bám lane đã ổn hay chưa.**

Mỗi bài test dưới đây có: mục đích, lệnh chạy, và **tiêu chí đạt/không đạt bằng số**.
Không có tiêu chí bằng số thì không phải test, chỉ là chạy thử.

> **Nguyên tắc:** thu xong buổi nào thì tối đó replay ở nhà và ra được số. Không
> để dồn ba buổi rồi mới phân tích — sai một tham số camera là mất cả ba buổi.

---

## Bối cảnh: vì sao T0 phải làm trước

Camera CSI của xe bị **lens shading màu**: đo trên `raw_camera.avi` (1485 frame),
tỉ lệ R/G ở tâm ảnh là `1.02` nhưng ở góc ảnh lên `1.59` — góc ảnh đỏ gấp ~1.6 lần
tâm ảnh. Bản đồ này đối xứng (trái `1.531` / phải `1.515`, trên `1.364` / dưới `1.337`),
nên nó là đặc tính ống kính, không phải do cảnh.

Hệ quả trực tiếp lên Bài 1: line giữa của sa bàn **màu đỏ**, phải tách bằng ngưỡng
màu. Nhưng một ngưỡng HSV duy nhất không thể dùng cho cả khung hình khi góc ảnh đỏ
gấp 1.6 lần tâm — chỉnh đủ bắt line ở giữa thì viền ảnh báo đỏ giả, chỉnh sạch viền
thì mất line ở giữa. **Mọi tham số ngưỡng tune trước khi sửa shading đều phải tune lại.**

Code đã có sẵn:

| Thành phần | Đường dẫn |
|---|---|
| Bộ sửa shading | [shading.py](src/jetracer_baseline/perception/shading.py) |
| Tool hiệu chuẩn | [calib_shading.py](tools/calib_shading.py) |
| Hệ số hiện tại (tạm) | `configs/shading.yaml` |
| Bật/tắt | `camera.shading.enabled` trong `configs/default.yaml` |

Hệ số đang có trong repo được ước lượng bằng `--mode drive` từ `raw_camera.avi`.
Nó giảm được 89% biên độ ám đỏ, nhưng **là tạm thời** vì hai lý do:

1. `--mode drive` lấy trung bình một video chạy thật, nội dung cảnh không triệt tiêu hết.
2. `raw_camera.avi` là **300×300**, còn `configs/default.yaml` capture **640×480** từ
   sensor mode 1280×720. Hệ số được chuẩn hoá theo bán kính nên vẫn dùng được **nếu
   góc nhìn giống nhau**, nhưng nếu 300×300 kia đến từ một crop khác thì hệ số sai.

T0 xoá cả hai rủi ro đó trong 10 phút.

---

# BUỔI 1 — Xe kê bánh, chưa chạy (~45 phút)

## T0 — Hiệu chuẩn shading bằng flat-field ⭐ làm đầu tiên

**Mục đích:** chốt hệ số sửa màu trên **đúng camera** và **đúng độ phân giải thi đấu**.

**Chuẩn bị:** một tờ A4 trắng, hoặc tấm foam trắng, hoặc mảng tường trắng phẳng.
Yêu cầu: **một màu, sáng đều, phủ kín khung hình**.

**Cách quay:**

1. Đặt xe dưới **đúng ánh sáng của sa bàn** (quan trọng — đừng quay ở phòng khác).
2. Giơ tờ giấy cách ống kính ~5–10 cm sao cho **phủ kín khung hình**, hơi mờ là tốt
   (ta cần trường phẳng, không cần chi tiết).
3. Di chuyển tờ giấy nhẹ nhàng trong lúc quay để triệt tiêu vết bẩn/nếp gấp.
4. Kiểm tra preview: không được có vùng cháy trắng (255) và không có bóng đổ rõ.

```bash
python3 tools/collect_dataset.py --mode video --source csi --session flatfield --out data/calib --seconds 15 --raw
```

> **`--raw` là bắt buộc ở bước này.** `camera.shading.apply_at: source` làm mọi
> frame ra khỏi camera đã được sửa màu sẵn. Hiệu chuẩn trên ảnh đã sửa = đo lens
> shading của một camera đã được bù shading → hệ số ra gần như vô hiệu. `--raw`
> bỏ qua lớp sửa và lấy đúng ảnh thô của cảm biến.

Copy video về PC rồi chạy:

```bash
python tools/calib_shading.py --mode flatfield --source data/calib/flatfield_<timestamp>.avi --preview reports/shading_flatfield.png
```

**Tiêu chí đạt:**

| Chỉ số trong output | Ngưỡng |
|---|---|
| Kiểm tra đối xứng trái/phải, trên/dưới | lệch **< 10%** (tool tự báo `DOI XUNG`) |
| Biên độ R/G tâm→góc **sau** khi sửa | **< 0.10** |
| Pixel chạm trần 255 | **< 1%** |

Tool **tự từ chối ghi file** nếu kiểm tra đối xứng thất bại. Nếu bị từ chối: ảnh
flat-field chưa đủ đều (bóng đổ, giấy không phủ kín, hoặc lóa một bên) — quay lại.

Mở `reports/shading_flatfield.png` xem ảnh trước/sau. Ảnh "SAU" phải là một mảng
xám/trắng **đều màu**, không còn hồng ở góc.

> **Ghi lại vào metadata session:** độ phân giải capture, `sensor_id`, `flip_method`,
> điều kiện sáng. Đổi bất kỳ thứ nào trong đó → hệ số shading **không còn hiệu lực**,
> phải hiệu chuẩn lại.

---

## T1 — Nghiệm thu shading trên sa bàn thật

**Mục đích:** xác nhận sau khi sửa, màu **đồng đều trên toàn khung hình** khi nhìn sa bàn.

Đặt xe trên sa bàn, hướng camera vào một đoạn có **line đỏ chạy từ giữa ra sát mép ảnh**
(đoạn cua là tốt nhất). Quay 20 giây, đẩy xe chậm bằng tay dọc đoạn đó.

```bash
python3 tools/collect_dataset.py --mode video --source csi --session t1_nghiemthu --out data/calib --seconds 20
```

Về PC, đo độ đồng đều của mask đỏ giữa viền và tâm:

```bash
python tools/calib_shading.py --mode drive --source data/calib/t1_nghiemthu_<ts>.avi --out /tmp/bo.yaml --preview reports/t1.png
```

**Tiêu chí đạt:** trong bảng "Ti le do duoc theo ban kinh", cột `R/G` từ bin đầu đến
bin cuối chênh nhau **< 0.15** (trước khi sửa, con số này là 0.57).

Nếu vẫn > 0.15: hệ số T0 chưa đúng cho ánh sáng sa bàn → quay lại T0 dưới đúng đèn sa bàn.

---

## T2 — Đo FPS thật trên Jetson ⭐ bằng chứng đang thiếu hoàn toàn

**Mục đích:** 10 điểm FPS của Bài 1 là **nhị phân** (≥ 20 FPS được 10, 19.9 được 0).
Hiện tại **toàn bộ 41 file log trong `logs/` đều là replay synthetic không realtime**
(0.1–0.8 giây, 200–690 FPS) — chưa có một log nào từ Jetson thật. Chưa đo thì chưa biết.

Xe **kê bánh khỏi mặt đất**, camera nhìn vào sa bàn:

```bash
# Chạy 60 giây trên xe, driver dryrun -> bánh không quay, vẫn đo đúng vòng điều khiển
python3 -m src.jetracer_baseline.cli run --task speed --driver dryrun --max-seconds 60 --record
```

Đọc dòng `FPS trung binh (ca luot)` trong output. Đây là con số đối chiếu ngưỡng 20
của BTC — nó đo **chu kỳ thực của vòng lặp**, không phải `1/thời-gian-xử-lý`.

**Chạy 3 lần**, ghi cả ba. Song song mở terminal thứ hai:

```bash
tegrastats --interval 1000 | tee reports/tegrastats_t2.txt
```

**Tiêu chí đạt:**

| Chỉ số | Ngưỡng |
|---|---|
| `FPS trung binh (ca luot)` | **≥ 24** cả 3 lượt (biên an toàn trên mốc 20) |
| `latency_ms` p95 trong CSV | **≤ 25 ms** |
| CPU trong tegrastats | không có core nào ghim 100% liên tục |

**Nếu FPS < 24:** đo trước, tối ưu sau. Hai chỗ đắt nhất theo thứ tự: (1) sửa shading
— đo được 1.6 ms/frame ở 320×240 trên PC dev, đường nhanh `apply_resized` đã resize
trước khi sửa để không trả giá gấp đôi; (2) `lane.process` — 2.2 ms/frame trên PC dev.
Tắt `camera.shading.enabled` rồi đo lại để biết shading chiếm bao nhiêu.

---

# BUỔI 2 — Đẩy tay / lái tay trên sa bàn (~60 phút)

Chưa cho xe tự chạy. Mục tiêu buổi này là **video**, không phải thành tích.

Dùng notebook `collect_drive.ipynb` (có gamepad + ghi session + metadata) hoặc
`tools/collect_dataset.py --mode video`.

## T3 — Video toàn vòng, tách theo session

Quay **mỗi điều kiện một session riêng**. Tên session sẽ thành tên file — sau này
chia train/val/test **phải chia theo session**, không random frame.

| Session | Nội dung | Thời lượng |
|---|---|---|
| `s1_sang_giua` | Đủ 1 vòng, xe giữ giữa lane | ≥ 2 vòng |
| `s2_sang_lechtrai` | Đủ 1 vòng, cố ý bám lệch trái | ≥ 1 vòng |
| `s3_sang_lechphai` | Đủ 1 vòng, cố ý bám lệch phải | ≥ 1 vòng |
| `s4_toi` | Vòng như s1 nhưng tắt bớt đèn / cuối ngày | ≥ 2 vòng |
| `s5_bongdo` | Vòng có người/vật tạo bóng đổ lên mặt đường | ≥ 1 vòng |

```bash
python3 tools/collect_dataset.py --mode video --source csi --session s1_sang_giua --out data/speed --seconds 180
```

**Vì sao cần cả lệch trái và lệch phải:** CTE chỉ có nghĩa nếu ta biết ground truth.
Ba session `giữa / lệch trái / lệch phải` cho phép kiểm tra **dấu và độ lớn của CTE
có đúng không** — thứ mà test hiện tại không kiểm tra được.

**Tiêu chí đạt:** mỗi session có ≥ 1 vòng liền mạch, không mất frame, không rung quá
mức. Xem lại video trước khi rời sa bàn.

## T4 — Ca khó và ca lỗi

Những đoạn này quyết định điểm số, đừng bỏ:

- Cua gắt nhất trên sa bàn — quay chậm, nhiều lượt.
- Đoạn line đỏ bị **đứt dài** hoặc mờ.
- Đoạn có **phản chiếu / lóa** trên mặt sàn bóng.
- Đoạn camera nhìn thấy nhiều **nền văn phòng** (người, ghế, bảng).
- Xe ở **sát mép lane**, sắp lệch ra ngoài.

```bash
python3 tools/collect_dataset.py --mode video --source csi --session s6_cakho --out data/speed --seconds 120
```

## T5 — Vạch xuất phát và 3 checkpoint

**Đây là 30 điểm của Bài 1 và hiện chưa có dòng code nào.** Không có video thì không
xây được event detector.

Với **từng** checkpoint (1, 2, 3) và vạch xuất phát/kết thúc, quay đoạn xe tiến tới —
đi qua — đi khỏi, ở tốc độ chậm:

```bash
python3 tools/collect_dataset.py --mode video --source csi --session s7_checkpoint1 --out data/speed --seconds 60
```

Chụp thêm **ảnh tĩnh cận cảnh** marker checkpoint và ghi chú vào metadata:
marker màu gì, hình gì, rộng bao nhiêu, có số in trên đó không, đặt trên mặt đường
hay dựng bên lề. Đây là thông tin quyết định chọn OpenCV hay detector.

**Tiêu chí đạt:** mỗi checkpoint có ≥ 3 lượt đi qua từ hướng thi đấu, thấy rõ marker
từ lúc mới xuất hiện trong khung hình đến lúc ra khỏi.

---

# Ở NHÀ — Phân tích (không cần xe)

## T6 — Replay và ra số

```bash
python -m src.jetracer_baseline.cli replay --task speed --source video \
    --video data/speed/s1_sang_giua_<ts>.avi --frames 100000 --log-dir logs/replay
```

```bash
python tools/analyze_log.py logs/replay/run_speed_<ts>.csv
```

**Đối chiếu mục tiêu §4.7 của tài liệu phương pháp:**

| Metric | Mục tiêu | Đo trên `raw_camera.avi` hiện tại |
|---|---|---|
| `cte_rms` | ≤ 0.15 | **0.211** ❌ |
| `abs_cte_p95` | ≤ 0.35 | **0.417** ❌ |
| `lane_loss_rate` | ≤ 2% | **0.0%** ❌ (giả — xem dưới) |
| steering đổi dấu | thấp | **4.8 lần/giây** ❌ |

`lane_loss_rate = 0%` **không phải điểm tốt**. Ngưỡng `lane.min_pixels = 60` trong khi
số pixel thực tế là 18.000–31.000, nên nhánh "mất line" **không bao giờ kích hoạt được**.
Khi replay data mới, kiểm tra thẳng con số này:

```bash
python -c "
import csv,sys,collections
rows=list(csv.DictReader(open(sys.argv[1])))
print(collections.Counter(r['event'] for r in rows))
" logs/replay/run_speed_<ts>.csv
```

Nếu `lane_lost` = 0 trên toàn bộ session `s6_cakho` thì detector đang **không có khả
năng báo lỗi**, không phải đang chạy hoàn hảo.

## T7 — Chốt lại ngưỡng màu (sau khi đã có shading đúng)

Ngưỡng HSV cho line đỏ **phải tune lại sau T0** — sửa shading làm giảm độ bão hoà đỏ
giả ở viền, nên ngưỡng cũ sẽ quá chặt. Đo trên `raw_camera.avi` sau khi sửa:

| Ngưỡng S | Số frame bắt được line đỏ trong ROI |
|---|---|
| `S > 60` | 99/99 |
| `S > 80` | 99/99 |
| `S > 100` | 99/99 |
| `S > 120` | 91/99 ← ngưỡng cũ, bắt đầu mất frame |
| `S > 140` | 75/99 |

Điểm xuất phát đề nghị: `H < 10 hoặc H > 170`, `S > 80`, `V > 70`. Chốt lại bằng data
thật của bạn, đừng lấy nguyên số này.

Dùng `tools/tune_lane.py` để xuất ảnh 4 ô từng bước xử lý khi dò ngưỡng.

---

# CÁCH CHẠY

## Ở nhà (không cần xe) — luôn chạy trước khi mang code lên xe

```bash
python tests/test_smoke.py
```

```bash
python tools/check_py36.py
```

```bash
python -m src.jetracer_baseline.cli replay --source synthetic --frames 200
```

Replay trên video sa bàn đã quay:

```bash
python -m src.jetracer_baseline.cli replay --task speed --source video --video raw_camera.avi --frames 100000 --log-dir logs/replay
```

Xem detector đang bám vào cái gì (ảnh 4 ô từng bước — mở ra xem trước khi tin số liệu):

```bash
python tools/tune_lane.py --source video --video raw_camera.avi --frames 6 --every 200 --out reports/lane
```

## Trên xe

Bước 1 — kiểm tra phần cứng, **bánh kê khỏi mặt đất**:

```bash
python3 tools/check_hardware.py --driver nvidia --camera-seconds 3
```

Bước 2 — hiệu chuẩn shading (T0 ở trên), rồi đo FPS thật, **vẫn kê bánh**:

```bash
python3 -m src.jetracer_baseline.cli run --task speed --driver dryrun --max-seconds 60 --record
```

`--driver dryrun` = bánh không quay nhưng vòng điều khiển chạy đầy đủ. Đọc dòng
`FPS trung binh (ca luot)` — đây là con số đối chiếu ngưỡng 20 của BTC.

Bước 3 — chạy thật, tốc độ thấp. Đặt xe **ngay trên vạch đứt**, người cầm sẵn nút dừng:

```bash
python3 -m src.jetracer_baseline.cli run --task speed --driver nvidia --max-seconds 60 --record
```

Bước 4 — lượt nhanh (chỉ đổi config, **không sửa code**):

```bash
python3 -m src.jetracer_baseline.cli run --task speed --driver nvidia --override configs/fast.yaml --record
```

## Ba núm chỉnh khi xe chạy chưa đúng

Chỉnh **một núm mỗi lần**, chạy lại, xem log. Đổi hai thứ cùng lúc thì không biết cái nào có tác dụng.

| Triệu chứng | Sửa trong `configs/default.yaml` | Hướng |
|---|---|---|
| Cắt cua muộn, ăn vào mép trong | `control.steer_lookahead_weight` | tăng (0.5 → 0.7) |
| Lắc qua lắc lại trên đường thẳng | `control.steer_lookahead_weight` | giảm (0.5 → 0.3) |
| Vẫn lắc | `control.pid.kp` | giảm (0.60 → 0.45) |
| Vào cua quá nhanh, văng ra | `control.curve_slowdown` | tăng (0.35 → 0.5) |
| Bò quá chậm ở đoạn thẳng | `control.curve_slowdown` | giảm, và tăng `control.v_max` |
| `n_bands` trong log thường ≤ 2 | `lane.min_blob_area` | giảm (bắt được nhiều nét đứt hơn) |
| Bám nhầm viền lane | `lane.max_run_frac` | giảm (0.45 → 0.30) |
| Đổi sang sa bàn thi | `lane.line_color` | `red` → `white` |

## Đọc log sau mỗi lượt

```bash
python tools/analyze_log.py logs/run_speed_<ts>.csv
```

Bốn cột mới cần nhìn: `lane_found`, `n_bands`, `curvature`, `throttle`.

- `n_bands` thường ≤ 2 → đang bám bằng rất ít dữ liệu, sắp mất vạch.
- `curvature` ghim ở ±1.0 → đa thức đang ngoại suy, ga sẽ tụt về `v_min` cả lượt.
- `throttle` bằng `v_min` gần như suốt → `curve_slowdown` hoặc `slowdown` quá cao.
- `lane_found` = 0 ở đâu, mở video đúng mốc thời gian đó ra xem.

---

# Thứ tự ưu tiên nếu thiếu thời gian

Làm theo đúng thứ tự này, đừng đảo:

1. **T0** — 10 phút, xoá bỏ một biến gây nhiễu cho mọi thí nghiệm sau.
2. **T2** — 5 phút, trả lời câu hỏi 10 điểm mà hiện chưa có bằng chứng nào.
3. **T3** (chỉ `s1_sang_giua`) — 10 phút, đủ để bắt đầu sửa lane detector ở nhà.
4. **T5** — 15 phút, không có video này thì 30 điểm checkpoint không làm được.
5. Phần còn lại của T3, rồi T4.

---

# Checklist mang theo

- [ ] Tờ A4 trắng / tấm foam trắng (cho T0)
- [ ] Thẻ nhớ / USB còn ≥ 8 GB
- [ ] Thước đo (ghi bề rộng lane và bề rộng marker checkpoint vào metadata)
- [ ] Điện thoại chụp ảnh **vị trí và góc gắn camera trên xe** — đổi góc camera là
      phải hiệu chuẩn lại cả shading lẫn warp
- [ ] Sổ ghi: mỗi session ghi giờ, điều kiện sáng, sự cố
