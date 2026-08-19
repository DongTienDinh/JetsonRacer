# Phương pháp, giải pháp kỹ thuật và công cụ cho Jetson AI Racer Challenge 2026

## 1. Mục đích tài liệu

Tài liệu này mô tả hướng nghiên cứu và triển khai cho hai bài thi **Speed Track** và **Smart City** theo bốn tiêu chí đánh giá Technical Paper:

1. Xác định rõ vấn đề, mục tiêu và câu hỏi nghiên cứu.
2. Mô tả pipeline, baseline, biến thí nghiệm và lý do chọn phương pháp.
3. Đánh giá định lượng bằng log, metric, bảng và biểu đồ.
4. So sánh phương pháp, phân tích ưu/nhược điểm và trade-off.

Đây là thiết kế đồng bộ với code hiện tại. Các thành phần được đánh dấu như sau:

- **Đã có:** đã tồn tại trong repository và có kiểm thử offline.
- **Cần hiệu chỉnh:** đã có code nhưng chưa chốt tham số trên sa bàn thật.
- **Cần triển khai:** mới có trong thiết kế hoặc mới có interface/stub.
- **Tùy chọn:** chỉ thực hiện khi baseline không đạt mục tiêu.

Không xem các ngưỡng mục tiêu trong tài liệu là kết quả đã đạt. Mọi kết quả đưa vào Technical Paper phải truy ngược được tới dataset, video và file log thực nghiệm.

**Nguồn yêu cầu hiện hành:** `Đề bài chi tiết (1).docx`, bản cập nhật ngày 18/08/2026. Bản này đã **loại bỏ chướng ngại vật khỏi Speed Track**; vì vậy Bài 1 trong tài liệu này không còn nhánh phát hiện/tránh obstacle hoặc YOLO obstacle.

---

## 2. Bài toán nghiên cứu tổng thể

### 2.1. Vấn đề

Xây dựng hệ thống xe tự hành chạy hoàn toàn trên JetRacer/Jetson Nano, chỉ sử dụng camera đơn và phần cứng do Ban tổ chức cung cấp, đáp ứng đồng thời:

- Bám lane ổn định, vượt đủ checkpoint và hoàn thành vòng Speed Track.
- Nhận diện biển báo, đèn giao thông và chọn đúng hướng ở Smart City.
- Pipeline điều khiển đạt trung bình tối thiểu 20 FPS khi thi Speed Track.
- Thời gian từ lúc biển/tín hiệu xuất hiện rõ đến lúc phát sinh lệnh tương ứng không quá 300 ms.
- Không phụ thuộc điều khiển tay trong lượt thi chính thức.
- Có log đủ để kiểm chứng FPS, latency, nhận diện và quyết định điều khiển.

### 2.2. Câu hỏi nghiên cứu

| Mã | Câu hỏi nghiên cứu |
|---|---|
| RQ1 | CV cổ điển hay mô hình học sâu cho lane đem lại trade-off tốt hơn giữa độ ổn định, FPS và khả năng hiệu chỉnh nhanh trên Jetson Nano? |
| RQ2 | Cặp tham số PID và profile tốc độ nào giảm CTE mà không gây dao động lái hoặc lệch lane? |
| RQ3 | Cơ chế start gate, checkpoint event và finish event nào đủ tin cậy mà không làm giảm FPS bám lane? |
| RQ4 | Model detector và kích thước input nào đạt recall biển/đèn cao trong giới hạn latency 300 ms? |
| RQ5 | Temporal voting, detection age và red latch giảm bao nhiêu quyết định sai so với quyết định từ một frame? |
| RQ6 | Kiến trúc hybrid `CV + detector + FSM` có ổn định hơn end-to-end `ảnh -> lệnh lái` khi ánh sáng và sa bàn thay đổi không? |

### 2.3. Giả thuyết chính

- **H1:** CV lane + PID đạt FPS và khả năng hiệu chỉnh tại chỗ tốt hơn CNN lane trong điều kiện dữ liệu sa bàn còn ít.
- **H2:** Adaptive throttle theo CTE và độ cong giảm số lần lệch lane ở cua so với tốc độ cố định.
- **H3:** Detector nhẹ TensorRT FP16 ở input 320 có thể đáp ứng latency 300 ms nếu chạy bất đồng bộ và chỉ giữ frame mới nhất.
- **H4:** Voting trên các inference độc lập kết hợp red latch làm giảm false-green và bỏ sót đèn đỏ so với quyết định một frame.
- **H5:** Chia dataset theo session cho kết quả test thực tế hơn chia ngẫu nhiên các frame liên tiếp.

---

## 3. Kiến trúc dùng chung

```text
CSI camera 30 FPS
       |
       v
Capture thread -- chỉ giữ frame mới nhất
       |
       +---------------------------+
       |                           |
       v                           v
Lane/stop-line CV             Detector TensorRT
20-30 Hz                      8-12 Hz
       |                           |
       +-------------+-------------+
                     v
             Temporal reasoning
                     |
                     v
               FSM / task logic
                     |
                     v
             PID / turn controller
                     |
                     v
       PCA9685 driver + watchdog + emergency stop
                     |
                     v
        CSV log + video + sidecar metadata
```

### 3.1. Nguyên tắc hệ thống

- Camera live dùng **latest-frame buffer**, không dùng queue dài.
- Lane, FSM và controller ưu tiên chạy 20-30 Hz.
- Detector chạy chậm hơn trong worker riêng và cache kết quả mới nhất.
- Temporal voting chỉ cập nhật khi có `inference_id` mới, không đếm lặp một inference ở nhiều control tick.
- Mỗi detection phải có `frame_id`, timestamp và tuổi kết quả `detection_age_ms`.
- Ghi ảnh/video chạy ở thread riêng; khi bộ nhớ đệm đầy thì bỏ mẫu thay vì chặn vòng điều khiển.
- Lệnh cuối cùng trước driver luôn qua hard limit, watchdog và emergency stop.

### 3.2. Công cụ dùng chung

| Công đoạn | Công cụ/thư viện | Nơi chạy | Trạng thái |
|---|---|---|---|
| Camera CSI | GStreamer, OpenCV | Jetson | Đã có |
| Xử lý ảnh | OpenCV, NumPy | Jetson + PC | Đã có |
| Điều khiển | PCA9685/ServoKit, PID | Jetson | Đã có, cần hiệu chỉnh |
| Thu dữ liệu | `collect_drive.ipynb`, `tools/collect_dataset.py` | Jetson | Đã có |
| Gán nhãn | CVAT, Label Studio hoặc Roboflow | PC/web | Chọn một công cụ chính |
| Train detector | PyTorch, Ultralytics | PC/Colab | Cần triển khai workflow |
| Augmentation | Albumentations hoặc augmentation của framework train | PC/Colab | Cần triển khai |
| Định dạng trung gian | ONNX | PC -> Jetson | Cần triển khai |
| Inference | TensorRT FP16 trực tiếp | Jetson | Cần triển khai backend |
| Log | CSV + video + sidecar + metadata JSON | Jetson | Đã có phần lớn |
| Phân tích | Python, NumPy, Matplotlib | PC | Đã có nền tảng |
| Theo dõi tài nguyên | `tegrastats` | Jetson | Cần đưa vào protocol test |

---

## 4. Bài 1 - Speed Track

## 4.1. Mục tiêu kỹ thuật

- Xuất phát đúng hiệu lệnh.
- Bám lane tối, line trắng đứt khúc ở giữa.
- Vượt ba checkpoint đúng thứ tự.
- Hoàn thành vòng và ghi nhận vạch kết thúc.
- Duy trì FPS pipeline trung bình >= 20.
- Ưu tiên 0 lần lệch lane và hoàn thành đủ checkpoint trước khi tối ưu thời gian.

## 4.2. Pipeline baseline

```text
Camera
-> resize 320x240
-> ROI mặt đường
-> cân bằng sáng + Gaussian blur
-> adaptive threshold
-> morphology
-> perspective transform
-> histogram/tâm line theo nhiều dải ảnh
-> CTE + độ cong xấp xỉ
-> EMA
-> PID
-> adaptive throttle
-> steering/throttle
```

Baseline chính là **CV cổ điển + PID**. Không dùng YOLO hoặc CNN để bám lane ở pha đầu.

## 4.3. Phương pháp và giải pháp kỹ thuật

| Thành phần | Phương pháp baseline | Cải tiến/fallback | Công cụ | Trạng thái |
|---|---|---|---|---|
| ROI | Crop phần dưới ảnh | ROI theo tốc độ | OpenCV | Đã có |
| Chuẩn hóa sáng | Equalize histogram | CLAHE, gamma/HSV-V normalization | OpenCV | Đã có baseline |
| Tách line | Adaptive threshold | HSV/colour threshold, Otsu preset | OpenCV | Đã có baseline |
| Khử nhiễu | Gaussian blur, morphology open | Morphology close, connected components | OpenCV | Đã có một phần |
| Bird's-eye view | Perspective transform | Hiệu chỉnh lại bốn điểm warp theo camera | OpenCV | Đã có, cần hiệu chỉnh |
| Tâm lane | Histogram/tâm pixel theo ba dải | Sliding windows, polynomial fitting | NumPy/OpenCV | Đã có baseline |
| Lọc theo thời gian | EMA | Kalman filter | NumPy | Đã có EMA |
| Mất line ngắn hạn | Giữ CTE cũ và giảm tốc | Optical flow hoặc predicted CTE | OpenCV | Có xử lý cơ bản |
| Điều khiển lái | PID theo CTE | Stanley/Pure Pursuit khi có lane geometry tốt | Code hiện tại | Đã có, cần tune |
| Điều khiển tốc độ | Giảm tốc theo `abs(CTE)` | Thêm curvature và lane confidence | Code hiện tại | Có một phần |
| Start/checkpoint/finish | Gate có chủ đích và event detector | Marker/line detector hoặc class riêng | OpenCV/YOLO | Cần triển khai |
| Phát hiện kẹt | Frame-difference motion estimator | Optical flow magnitude | OpenCV | Đã có, cần hiệu chỉnh |
| Thoát kẹt | Lùi ngắn + đánh lái có giới hạn | Recovery theo lịch sử hướng | FSM | Đã có baseline |

## 4.4. Start gate, checkpoint và finish

Đề cập nhật không yêu cầu Speed Track phát hiện hoặc tránh chướng ngại vật. Các sự kiện cần quan tâm là xuất phát đúng hiệu lệnh, vượt checkpoint 1-2-3 đúng thứ tự và quay lại vạch kết thúc.

### Start gate

```text
Khởi tạo camera + pipeline + driver
-> giữ FSM ở IDLE, throttle = 0
-> nhận hành động start được BTC cho phép
-> ghi event=start và bắt đầu tính t_rel
-> chuyển sang LANE_FOLLOW
```

Không dùng tay cầm/joystick để điều khiển trong lượt thi. Cách start cụ thể phải được BTC xác nhận: khởi chạy chương trình đúng lúc có hiệu lệnh, một start gate phần mềm được phép, hoặc nhận diện tín hiệu xuất phát nếu BTC yêu cầu hoàn toàn tự động từ trước hiệu lệnh.

### Checkpoint và finish event

Xe không bắt buộc phải phân loại số checkpoint để bám lane, nhưng event detector có ích để:

- Xác nhận checkpoint 1-2-3 theo thứ tự trong log.
- Không dừng nhầm ở checkpoint trung gian.
- Phát hiện lần quay lại vạch xuất phát/kết thúc sau khi đã đi đủ checkpoint.
- Gọi trạng thái `FINISHED` và cắt throttle có chủ đích.

Baseline nên ưu tiên xử lý ảnh nhẹ:

```text
ROI marker/checkpoint
-> threshold màu/sáng hoặc template hình học
-> temporal confirmation
-> checkpoint_index tăng đúng thứ tự
-> finish chỉ hợp lệ khi checkpoint_index = 3
```

Chỉ dùng detector học sâu cho checkpoint/finish nếu marker thực tế không thể tách ổn định bằng OpenCV.

### Công thức điểm Speed Track theo đề cập nhật

```text
Score = max(0, checkpoint_score + fps_score + time_score - penalty)
checkpoint_score = số checkpoint hợp lệ * 10, tối đa 30
fps_score = 10 nếu FPS trung bình >= 20, ngược lại 0
time_score = 60 - 2 * floor(thời_gian / 10), không âm
```

Các lỗi còn lại trong Bài 1:

- Lệch lane: `-10 điểm + 15 giây/lần`.
- Xuất phát sớm: `-10 điểm + 10 giây/lần`.
- Đi ngược chiều: hủy lượt.
- Can thiệp thủ công khi đang chạy: hủy lượt, trừ thao tác được BTC cho phép.

Ví dụ chính thức của đề mới:

```text
3 checkpoint = 30
FPS 22 = 10
hoàn thành 100 giây = 40
lệch lane 1 lần = -10
Tổng = 70 điểm
```

## 4.5. Thu thập dữ liệu Speed Track

Thu theo từng session độc lập:

- Đường thẳng, cua nhẹ, cua gắt.
- Xe ở giữa, lệch trái, lệch phải.
- Tốc độ thấp, trung bình và profile nhanh.
- Sáng đều, tối, bóng đổ, phản chiếu.
- Line rõ, mờ, đứt, bị che một phần.
- Các frame trước, tại và sau checkpoint/vạch kết thúc.
- Các ca lỗi: mất line, dao động lái, nhận nhầm checkpoint/finish và xe kẹt.

Tần suất lưu đề xuất:

- Video replay: 15-30 FPS.
- Ảnh làm dataset lane/checkpoint: 2-5 FPS sau khi loại gần trùng.
- Imitation-learning tùy chọn: 15-20 FPS kèm steering/throttle đồng bộ.

## 4.6. Thí nghiệm Speed Track

| Mã | Thí nghiệm | Biến độc lập | Biến kiểm soát | Metric quyết định |
|---|---|---|---|---|
| S1 | Threshold lane | block size, C, preset sáng | Cùng video | lane-loss rate, CTE RMS |
| S2 | ROI/warp | `roi_top`, bốn điểm warp | Cùng video, PID | CTE RMS, curvature noise |
| S3 | PID grid | Kp, Ki, Kd | Cùng video/tốc độ | CTE RMS, steering std, overshoot |
| S4 | Speed profile | `v_max`, `v_min`, slowdown | Cùng lane/PID | lap time, departure/lap, score mô phỏng |
| S5 | Checkpoint/finish event | threshold, template, detector tùy chọn | Cùng video toàn vòng | event accuracy, false finish, latency |
| S6 | Robustness | augmentation/preset sáng | Cùng pipeline | độ suy giảm CTE/lane-loss giữa điều kiện sáng |

Mỗi cấu hình cuối nên được chạy tối thiểu năm lượt độc lập trên xe; báo cáo mean, standard deviation và p95 thay vì chỉ báo cáo lượt tốt nhất.

## 4.7. Metric Speed Track

| Nhóm | Metric | Mục tiêu ban đầu |
|---|---|---|
| Lane | `cte_rms` | <= 0.15 |
| Lane | `abs_cte_p95` | <= 0.35 |
| Lane | `lane_loss_rate` | <= 2% frame |
| Điều khiển | `steering_std`, số lần đổi dấu/giây | Càng thấp càng tốt nhưng không làm tăng CTE |
| Nhiệm vụ | lane departures/lap | 0 |
| Nhiệm vụ | checkpoint pass | 3/3 đúng thứ tự |
| Nhiệm vụ | false start | 0 |
| Nhiệm vụ | finish event | Chỉ xuất hiện sau checkpoint 3 |
| Hệ thống | `fps_mean` | >= 20 FPS trên Jetson thật |
| Hệ thống | latency p95 | Không làm control loop tụt dưới 20 FPS |
| Điểm | `score_sim` | Tính từ log và công thức BTC |

## 4.8. Lập luận lựa chọn

| Phương pháp | Ưu điểm | Nhược điểm | Quyết định |
|---|---|---|---|
| CV lane | Nhẹ, dễ debug, tune tại chỗ | Nhạy sáng/màu | Baseline chính |
| Lane segmentation | Robust hình dạng hơn | Cần mask, train và TensorRT | Fallback khi CV thất bại |
| End-to-end steering | Pipeline ngắn | Khó giải thích, khó bảo đảm an toàn | Không ưu tiên |
| PID | Đơn giản, có thể giải thích | Cần tune theo tốc độ | Baseline chính |
| Stanley/Pure Pursuit | Dùng geometry rõ ràng | Cần đường lane ổn định | Thí nghiệm tùy chọn |
| Checkpoint bằng OpenCV | Nhẹ, dễ log event | Phụ thuộc marker thực tế | Baseline sau khi khảo sát marker |
| Detector checkpoint/finish | Robust hơn khi marker phức tạp | Tăng dữ liệu và latency | Chỉ dùng nếu OpenCV không đạt |

---

## 5. Bài 2 - Smart City

## 5.1. Mục tiêu kỹ thuật

- Nhận diện đúng biển báo lệnh và biển cấm.
- Phân biệt đèn đỏ và đèn xanh với ngưỡng an toàn bất đối xứng.
- Dừng trước vạch khi đỏ và chỉ đi khi xanh đã ổn định.
- Chọn đúng hướng tại mỗi giao lộ.
- Ghi nhớ tiến trình route và đi đến đúng vùng kết thúc.
- Phát sinh quyết định tương ứng trong tối đa 300 ms kể từ lúc tín hiệu xuất hiện rõ.

## 5.2. Pipeline Smart City

```text
Camera latest frame
   |-- OpenCV lane -> CTE
   |-- OpenCV stop-line -> found + distance
   `-- TensorRT detector -> signs/lights + bbox + inference metadata
                                |
                                v
                 temporal voting + detection age
                                |
                red latch + sign latch + route state
                                |
                                v
        CRUISE -> APPROACH -> WAIT_RED / DECIDE
              -> TURN -> REACQUIRE_LANE -> CRUISE
                                |
                                v
                       PID/turn controller
```

## 5.3. Các lớp nhận diện

Model Smart City đề xuất có tám class:

```text
turn_left
turn_right
go_straight
no_left
no_right
no_straight
red_light
green_light
```

`stop_line`, `intersection` và `finish` chỉ thêm thành class nếu giải pháp CV/route state không đủ ổn định. Không thêm class chỉ vì có thể gán nhãn; mỗi class phải phục vụ một quyết định cụ thể của FSM.

## 5.4. Phương pháp và giải pháp kỹ thuật

| Thành phần | Phương pháp baseline | Cải tiến/fallback | Công cụ | Trạng thái |
|---|---|---|---|---|
| Lane | Cùng CV lane Speed Track, tốc độ thấp | Segmentation nhẹ | OpenCV/TensorRT | Đã có baseline |
| Stop line | Otsu + row-fill vùng dưới ảnh | Hough line/class detector | OpenCV/YOLO | Đã có, cần tune |
| Biển/đèn | Detector nhẹ, input khoảng 320 | So sánh input 256/320/416 và model nano/small | PyTorch/Ultralytics | Cần train |
| Deploy | ONNX -> TensorRT FP16 | INT8 khi có calibration set đủ tốt | ONNX/TensorRT | Cần triển khai |
| Temporal voting | k-of-n trên inference độc lập | Weighted voting theo confidence/age | Code tracker | Cần sửa interface |
| Đèn đỏ | Ngưỡng recall cao + red latch | ROI màu HSV hỗ trợ | FSM/OpenCV | Cần hoàn thiện |
| Đèn xanh | Confidence cao + nhiều frame xác nhận | HSV verification | FSM/OpenCV | Có ý tưởng, cần kiểm chứng |
| Detection cũ | `detection_age_ms` quá ngưỡng -> không dùng | Near stop-line -> dừng mặc định | FSM | Cần triển khai |
| Biển lệnh | Latch một lần tại giao lộ | Bbox area/distance gate | FSM | Có baseline |
| Biển cấm | Chọn hướng hợp lệ từ route graph | Fallback config chỉ dùng khi map đã biết | FSM/route state | Cần triển khai |
| Rẽ | Steering override theo thời gian | Reacquire lane theo confidence | FSM + PID | Có baseline, cần cải tiến |
| Đích | Route step + finish detector | Marker/class finish | FSM/OpenCV/YOLO | Cần triển khai |

## 5.5. Temporal reasoning an toàn

Mỗi inference mới tạo một record:

```text
inference_id
source_frame_id
capture_timestamp
inference_timestamp
detections[]
```

Quy tắc khởi điểm để đem đi thực nghiệm, không coi là hằng số cuối cùng:

- Biển báo: ổn định khi xuất hiện trong `3/5` hoặc `4/6` inference mới.
- Đèn đỏ: ưu tiên recall, có thể latch sớm ở `2/4` inference với confidence phù hợp.
- Đèn xanh: ưu tiên precision, chỉ release red latch khi xanh ổn định, ví dụ `4/5` inference.
- Không đếm lặp cùng `inference_id`.
- Nếu detection quá cũ hoặc tín hiệu không chắc chắn khi gần stop line: `force_stop=True`.
- Red latch chỉ được xóa bởi green confirmation hoặc reset có chủ đích khi bắt đầu lượt mới.
- Sau khi thực hiện xong một biển tại giao lộ, xóa latch của biển đó để tránh rẽ lặp lại.

## 5.6. FSM và route state

### Trạng thái đề xuất

```text
IDLE
CRUISE
APPROACH_INTERSECTION
WAIT_RED
DECIDE
TURNING
REACQUIRE_LANE
FINISHED
FAIL_SAFE_STOP
```

### Dữ liệu trạng thái cần lưu

```text
route_step
intersection_id hoặc intersection_count
pending_instruction
last_processed_sign
red_latched
last_detection_age_ms
turn_start_time
lane_reacquired_count
```

Đối với biển cấm, không dùng một fallback cứng cho mọi giao lộ. FSM cần tham chiếu route graph hoặc bảng route đã xác định từ bố trí sa bàn:

```text
(intersection, heading, prohibited_direction)
-> valid_direction
-> next_intersection
```

## 5.7. Pipeline train và deploy detector

```text
Ảnh session thật
-> gán bbox
-> kiểm tra chất lượng nhãn
-> split theo session
-> augmentation train-only
-> fine-tune pretrained model trên PC/Colab
-> đánh giá test set cố định
-> export ONNX
-> kiểm tra ONNX output
-> chuyển ONNX sang TensorRT trên đúng Jetson
-> đánh giá lại accuracy/recall/latency trên Jetson
```

### Ràng buộc phiên bản

- Ultralytics dùng ở PC/Colab để train và export, không phải runtime bắt buộc trên Jetson.
- Jetson đang dùng Python 3.6 nên backend hiện tại không được phụ thuộc `from ultralytics import YOLO`.
- Cần triển khai `TensorRTDetector` gọi TensorRT trực tiếp.
- TensorRT engine phải được build trên đúng Jetson/JetPack/TensorRT mục tiêu.
- Trước khi export phải xác nhận chính xác xe chạy JetPack 4.5.1 hay 4.6 và phiên bản TensorRT thực tế.
- ONNX là artifact trao đổi; `.engine` là artifact phụ thuộc thiết bị/runtime.

## 5.8. Thu thập dữ liệu Smart City

Mỗi class cần có:

- Xa, vừa, gần.
- Góc nhìn trái, chính diện, phải.
- Góc nghiêng và biến dạng perspective.
- Sáng, tối, bóng đổ, ngược sáng.
- Rõ, motion blur, che khuất một phần.
- Background dễ gây nhầm.
- Ảnh âm tính không chứa biển/đèn.
- Ảnh tại vị trí xe bắt đầu phải ra quyết định, không chỉ ảnh cận cảnh đẹp.

Mục tiêu ban đầu:

- Tổng tối thiểu khoảng 1.500 ảnh đã kiểm tra nhãn.
- Tối thiểu khoảng 150 instance/class trước augmentation.
- Đèn đỏ và xanh phải có nhiều chuỗi video liên tục để đánh giá temporal logic.
- Tách train/validation/test theo session hoặc buổi thu, không chia ngẫu nhiên frame.

## 5.9. Augmentation

Chỉ áp dụng augmentation cho tập train:

- Brightness, contrast, gamma.
- HSV/white-balance shift vừa phải.
- Bóng đổ nhân tạo.
- Gaussian noise và motion blur.
- Scale, translate và perspective nhẹ.
- Che khuất một phần.
- Copy-paste có kiểm soát nếu giữ đúng tỷ lệ và phối cảnh.

Không áp dụng augmentation làm thay đổi nghĩa nhãn, ví dụ lật ngang ảnh biển rẽ trái thành rẽ phải mà không đổi label.

## 5.10. Thí nghiệm Smart City

| Mã | Thí nghiệm | Biến độc lập | Biến kiểm soát | Metric quyết định |
|---|---|---|---|---|
| M1 | Model/input | nano/small; 256/320/416 | Cùng dataset split | mAP, recall/class, latency, FPS |
| M2 | Precision | FP32, FP16, tùy chọn INT8 | Cùng model/input | accuracy delta, latency, memory |
| M3 | Augmentation | none/basic/strong | Cùng seed/model | mAP theo điều kiện sáng |
| M4 | Voting | 1 frame, 3/5, 4/6, 5/7 | Cùng detection sequence | intersection accuracy, delay |
| M5 | Red/green threshold | confidence và k/n | Cùng video đèn | red recall, green precision |
| M6 | Stop line | threshold/ROI/row-fill | Cùng video | distance error, stop success |
| M7 | End-to-end | FSM thường và FSM fail-safe | Cùng route | success rate, wrong route, red violation |

## 5.11. Metric Smart City

| Nhóm | Metric | Mục tiêu ban đầu |
|---|---|---|
| Detector | mAP@0.5 | >= 0.85 trên test tự thu |
| Detector | precision/recall từng class | Báo cáo đầy đủ, không chỉ mAP tổng |
| An toàn | red recall | Hướng tới 1.00 trên test an toàn |
| An toàn | green precision | Hướng tới >= 0.99 |
| Quyết định | intersection accuracy | 100% trên route đánh giá |
| Hệ thống | sign latency p95 | <= 300 ms trên Jetson |
| Hệ thống | detector actual rate | Báo cáo inference/s thật, không dùng chỉ số chỉ tính thời gian infer |
| Nhiệm vụ | success rate | >= 90% trước ngày thi |
| Nhiệm vụ | red-light violations | 0 |
| Nhiệm vụ | wrong-route runs | 0 ở cấu hình chốt |

## 5.12. Lập luận lựa chọn

| Phương pháp | Ưu điểm | Nhược điểm | Quyết định |
|---|---|---|---|
| YOLO nhẹ | Một model cho nhiều biển/đèn, bbox rõ | Cần dữ liệu và deploy TensorRT | Dùng cho Smart City |
| SSD-MobileNet | Phù hợp edge, có thể dễ tích hợp TensorRT cũ | Tool train ít thống nhất với repo | Ứng viên so sánh nếu YOLO export lỗi |
| HSV đèn | Rất nhẹ, dễ giải thích | Nhạy ánh sáng/background | Dùng như tín hiệu xác minh, không nên là detector duy nhất |
| Temporal voting | Giảm nhiễu một frame | Tăng delay | Bắt buộc, chọn k/n bằng thí nghiệm |
| FSM luật | Giải thích và kiểm thử được | Phụ thuộc route, nhiều edge case | Baseline chính |
| End-to-end policy | Có thể học hành vi phức tạp | Khó chứng minh an toàn và cần nhiều data | Không ưu tiên |

---

## 6. Quản lý dataset và vòng lặp cải tiến

## 6.1. Cấu trúc session đề xuất

```text
data/
  raw/
    speed/
      session_YYYYMMDD_condition/
    smartcity/
      session_YYYYMMDD_condition/
  annotations/
  manifests/
    train.txt
    val.txt
    test.txt
  processed/
  hard_examples/
  models/
```

Mỗi session cần metadata:

```text
camera resolution/FPS
Jetson/JetPack version
điều kiện sáng
tên sa bàn
vị trí camera
driver/calibration config
người thu
thời gian
ghi chú lỗi
```

## 6.2. Chia dữ liệu

Chọn một quy ước thống nhất cho toàn dự án:

```text
Train      70%
Validation 15%
Test       15%
```

Tỷ lệ chỉ là mục tiêu. Ràng buộc quan trọng hơn là toàn bộ frame của một session phải nằm trong cùng một tập. Nếu số session ít, ưu tiên tách theo điều kiện sáng/ngày quay thay vì ép đúng phần trăm.

## 6.3. Loại dữ liệu trùng và lỗi

- Sampling theo thời gian 2-5 FPS chỉ là bước đầu, chưa bảo đảm hết ảnh trùng.
- Dùng perceptual hash, SSIM hoặc embedding similarity để tạo danh sách ảnh gần trùng.
- Không xóa tự động mọi ảnh gần trùng; giữ lại chuỗi liên tục cần cho temporal evaluation.
- Kiểm tra ảnh mờ quá mức, frame đen, camera lỗi và nhãn ngoài ảnh.
- Kiểm tra class distribution và kích thước bbox theo class.

## 6.4. Hard-example mining

Sau mỗi lượt replay hoặc chạy xe:

```text
model inference
-> lọc false positive/false negative/low confidence
-> lấy frame trước và sau lỗi
-> gán nhãn lại
-> đưa vào hard_examples
-> train vòng tiếp theo
```

Các lỗi cần ưu tiên:

- Bỏ sót đèn đỏ.
- Nhận nhầm đỏ thành xanh hoặc xanh thành đỏ.
- Nhầm biển lệnh và biển cấm cùng hướng.
- Phát hiện biển ở background nhưng không thuộc giao lộ hiện tại.
- Mất line tại cua/bóng đổ.
- Nhận nhầm checkpoint trung gian thành vạch kết thúc.

## 6.5. Active learning

Chỉ gán nhãn có chọn lọc các mẫu:

- Confidence nằm gần threshold.
- Nhiều model/checkpoint không đồng thuận.
- Temporal sequence thay đổi nhãn liên tục.
- Điều kiện sáng hoặc góc nhìn chưa có đủ trong dataset.

Active learning là bước tối ưu chi phí gán nhãn, không thay thế test set cố định.

---

## 7. Log và phân tích định lượng

## 7.1. Trường log bắt buộc

Các trường code hiện tại đã có phần lớn:

```text
timestamp
t_rel
frame_id
fps
det_fps
latency_ms
sign_latency_ms
detected_object
confidence
decision
control_output
cte
state
event
```

Đề cập nhật khuyến nghị tên trường `sign`, trong khi code hiện tại dùng `detected_object`. Khi mở rộng schema cần chọn một trong hai cách: đổi tên có migration rõ ràng, hoặc giữ `detected_object` và thêm alias `sign`. Không để hai tên mang ý nghĩa khác nhau trong các session thực nghiệm.

Các trường nên bổ sung cho detector/FSM mới:

```text
inference_id
source_frame_id
detection_timestamp
detection_age_ms
bbox
red_latched
route_step
intersection_id
stopline_found
stopline_distance
lane_found
curvature
cpu_temp
gpu_load
memory_used
```

## 7.2. Nguyên tắc đo

- `fps_mean` của lượt chạy phải đo trên Jetson thật, không dùng FPS replay không realtime.
- `det_fps` phải là số inference hoàn tất trên thời gian thực, có tính cả chu kỳ chờ; có thể log thêm `inference_ms` riêng.
- `sign_latency_ms` phải đo cho từng lần xuất hiện biển/tín hiệu, không chỉ lần đầu tiên của mỗi class trong cả lượt.
- Timestamp capture, inference và command cần dùng cùng một clock monotonic nếu có thể.
- Mỗi bảng/biểu đồ trong paper phải ghi dataset split, số lượt lặp và cấu hình tương ứng.

## 7.3. Biểu đồ cho Technical Paper

- Timeline CTE, steering và throttle của một lượt.
- Heatmap `Kp x Kd` theo CTE RMS.
- Đường `v_max -> score_sim/departures/lap`.
- Pareto `mAP/recall -> latency/FPS` của model/input.
- Confusion matrix detector.
- Precision-recall curve từng class quan trọng.
- Boxplot latency trước/sau TensorRT.
- Bar chart ablation voting/red latch.
- Bảng failure cases kèm frame minh họa và giải thích.

---

## 8. Ánh xạ với code hiện tại

| Thành phần | File hiện tại | Trạng thái/hành động |
|---|---|---|
| Camera/latest frame | `src/jetracer_baseline/camera.py` | Đã có |
| Lane CV | `src/jetracer_baseline/perception/lane.py` | Đã có, tune sa bàn |
| Stop line | `src/jetracer_baseline/perception/stopline.py` | Đã có, tune và test |
| Detector stub/Ultralytics | `src/jetracer_baseline/perception/signs.py` | Thay runtime Jetson bằng `TensorRTDetector` |
| Danh sách class | `configs/default.yaml` | Bỏ class `obstacle`; giữ tám class Smart City |
| Temporal tracker | `src/jetracer_baseline/perception/signs.py` | Thêm `inference_id`, age và reset theo event |
| FSM | `src/jetracer_baseline/fsm.py` | Thêm red latch, route, finish, fail-safe |
| Main pipeline | `src/jetracer_baseline/pipeline.py` | Truyền metadata detector và log mới |
| PID | `src/jetracer_baseline/control/pid.py` | Đã có, tune |
| Driver/watchdog | `src/jetracer_baseline/control/driver.py` | Đã có, xác nhận xe thật |
| Thu lái tay | `src/jetracer_baseline/manual_collection.py` | Đã có |
| Thu ảnh/video | `tools/collect_dataset.py` | Đã có, bổ sung manifest/QC |
| Log | `src/jetracer_baseline/logging_csv.py` | Đã có, mở rộng schema trước khi thu số liệu chính thức |
| Phân tích | `tools/analyze_log.py` | Đã có nền tảng, bổ sung biểu đồ/ablation |
| Train model | Chưa có | Cần script/config train có seed và dataset manifest |
| Export ONNX | Chưa có | Cần script kiểm tra output parity |
| Build TensorRT | Chưa có | Cần protocol chạy trên đúng Jetson |
| Checkpoint/finish | Chưa có | Cần module/event detector |

---

## 9. Thứ tự triển khai

### P0 - Chốt phần cứng và môi trường

1. Xác nhận JetPack, CUDA, TensorRT, Python và OpenCV thực tế.
2. Xác nhận I2C `0x40`, channel lái/ga và nguồn xe.
3. Hiệu chỉnh steering center, hard limit, throttle deadzone.
4. Kiểm chứng emergency stop và watchdog.
5. Chạy tay ổn định và ghi đủ video/metadata.

### P1 - Speed Track baseline

1. Thu video đủ điều kiện sáng/cua.
2. Tune ROI, warp, threshold.
3. Tune PID trên replay, sau đó xác nhận trên xe.
4. Chốt safe profile trước, fast profile sau.
5. Bổ sung start/checkpoint/finish event.

### P2 - Hoàn thiện Speed Track và bằng chứng log

1. Chốt cơ chế start gate được BTC cho phép.
2. Thu video checkpoint/vạch kết thúc trên sa bàn thật.
3. Xây event detector nhẹ và kiểm tra đúng thứ tự 1-2-3.
4. Chỉ cho phép finish sau khi đủ ba checkpoint.
5. Kiểm chứng công thức điểm mới bằng log của một lượt hoàn chỉnh.

### P3 - Smart City perception

1. Thu/gán nhãn tám class.
2. Chốt split theo session.
3. Train và benchmark model/input.
4. Export ONNX, build TensorRT trên Jetson.
5. Đánh giá lại recall/precision và latency sau deploy.

### P4 - Smart City decision

1. Sửa voting theo inference mới.
2. Thêm detection age và fail-safe stop.
3. Thêm red latch/green release.
4. Thêm route state và finish state.
5. Replay video giao lộ trước khi chạy xe.

### P5 - Thực nghiệm và paper

1. Khóa code/config/model/dataset version.
2. Chạy đủ lượt lặp cho từng thí nghiệm.
3. Phân tích mean, std, p95 và failure cases.
4. Sinh bảng/biểu đồ trực tiếp từ log.
5. Viết Discussion, Limitation và trade-off dựa trên kết quả thật.

---

## 10. Checklist đáp ứng tiêu chí Technical Paper

### Xác định câu hỏi hoặc mục tiêu kỹ thuật

- [ ] Problem statement nêu rõ hai bài thi và giới hạn Jetson/camera.
- [ ] Có RQ/Hypothesis đo được, không chỉ mô tả tính năng.
- [ ] Phân biệt mục tiêu kỹ thuật với kết quả đã đạt.

### Phương pháp thiết kế thí nghiệm

- [ ] Có pipeline và baseline cho từng bài.
- [ ] Có biến độc lập, biến kiểm soát và số lượt lặp.
- [ ] Dataset split theo session và có test set cố định.
- [ ] Có ablation cho voting, augmentation hoặc speed control.

### Phân tích định lượng qua log dữ liệu

- [ ] FPS và latency đo trên Jetson thật.
- [ ] Báo cáo mean, std, p95 và số mẫu/lượt.
- [ ] Có confusion matrix và metric từng class.
- [ ] Có timeline CTE/steering/throttle và failure analysis.
- [ ] Mọi con số truy ngược được tới log/model/config cụ thể.

### Lập luận lựa chọn và so sánh mô hình

- [ ] So sánh CV lane với ít nhất một phương án học sâu hoặc giải thích rõ vì sao chưa dùng.
- [ ] So sánh model/input/precision bằng bảng Pareto.
- [ ] Giải thích trade-off accuracy, latency, FPS, bộ nhớ và khả năng debug.
- [ ] Nêu hạn chế: camera mono, domain shift, route luật cứng và dữ liệu sa bàn hạn chế.

---

## 11. Kết luận lựa chọn kỹ thuật

```text
Speed Track
= OpenCV lane + PID + adaptive speed
+ start/checkpoint/finish event
```

```text
Smart City
= OpenCV lane + stop-line
+ YOLO detector nhẹ train trên PC
+ ONNX -> TensorRT build trên Jetson
+ temporal voting + detection age
+ red latch + fail-safe FSM + route state
```

Kiến trúc hybrid này được chọn vì phù hợp giới hạn Jetson Nano, có thể giải thích và kiểm thử từng thành phần, đồng thời cho phép cải tiến có kiểm soát mà không phải thay toàn bộ hệ thống.
