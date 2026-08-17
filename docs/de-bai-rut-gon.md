# Đề bài rút gọn — checklist kiểm tra được

Rút từ `Đề bài chi tiết.docx.pdf` + `Thể lệ.docx.pdf`. Mỗi dòng là một yêu cầu **kiểm chứng được**: tick khi có bằng chứng (log/video/test), không tick theo cảm tính.

---

## A. Ràng buộc chung (áp dụng cả 2 bài)

- [ ] Không sửa phần cứng xe so với cấu hình BTC (camera, board, motor, servo, khung, pin) — ĐB §2.1
- [ ] Không can thiệp hệ thống đếm giờ / cảm biến / đèn hiệu / sa bàn / thiết bị chấm điểm — ĐB §2.1
- [ ] Trong lượt thi: xe **tự vận hành 100%**, không joystick/keyboard/remote — ĐB §2.2
- [ ] Có cơ chế xuất log `.txt`/`.csv` với các trường quy định — ĐB §7
- [ ] Có sẵn quy trình khởi động chương trình trong ≤ 10 phút bàn giao + 5 phút trên sa bàn — ĐB §2.3
- [ ] Tổng thi 25 phút: Speed Track ≤ 15 phút (3 lượt × 5), Smart City ≤ 10 phút (2 lượt × 5) — ĐB §2.3

**Định nghĩa cần thuộc lòng (ĐB §2.4):**

| Thuật ngữ | Định nghĩa chính xác |
|---|---|
| Lệch khỏi lane | **Hai bánh cùng một bên** vượt ra khỏi mép lane (không phải 1 bánh) |
| Checkpoint hợp lệ | **Toàn bộ 4 bánh** vượt vạch, **đúng chiều**, **không bỏ qua checkpoint trước** |
| Thời gian hoàn thành | Từ lúc xuất phát hợp lệ đến khi hoàn thành, **đã cộng thời gian phạt** |
| Hủy kết quả | Lượt tính 0 điểm hoặc không dùng để xếp hạng |

---

## B. Speed Track (30%)

**Sa bàn:** lane màu tối, line trắng **đứt khúc** ở giữa, 03 checkpoint đánh số, có chướng ngại vật (số lượng/vị trí công bố tại briefing), vạch xuất phát = vạch kết thúc.

### Nhiệm vụ

- [ ] **Xuất phát đúng hiệu lệnh** — chỉ di chuyển sau hiệu lệnh, vượt vạch đúng hướng
- [ ] **Bám lane** — không có 2 bánh cùng bên ra khỏi mép lane
- [ ] **Vượt 3 checkpoint đúng thứ tự** — cả 4 bánh, đúng chiều
- [ ] **Tránh vật cản** — không chạm/đẩy/kéo/làm dịch chuyển
- [ ] **Hoàn thành vòng** — đủ checkpoint + về vạch kết thúc trong 5 phút
- [ ] **Ghi log hiệu năng** — FPS trung bình + sự kiện điều khiển, hiển thị được khi BTC yêu cầu

### Điểm

```
Điểm lượt = max(0, CP + FPS + Time − Penalty)
CP   = n_checkpoint × 10, max 30      ⚠ mâu thuẫn với ví dụ §8 (xem BASELINE §6.1)
FPS  = 10 nếu FPS_tb ≥ 20, else 0
Time = 60 − 2 × (t/10), max 60
Điểm cuối = MAX của 3 lượt
```

### Lỗi

| Vi phạm | Trừ | Phạt | Xử lý |
|---|---|---|---|
| Đụng vật cản | −5/lần | +10 s | Chạy tiếp nếu còn di chuyển được |
| Lệch đường đua | −10/lần | +15 s | BTC đưa về checkpoint gần nhất, **đồng hồ dừng** khi BTC can thiệp |
| Xuất phát sớm | −10/lần | +10 s | Xuất phát lại nếu chưa đi xa |
| **Đi ngược chiều** | **hủy lượt** | — | Không tính điểm |
| **Can thiệp thủ công** | **hủy lượt** | — | Không tính điểm |

---

## C. Smart City (40%)

**Sa bàn:** mạng đường đô thị — giao lộ = đỉnh, đoạn đường = cạnh. Một số đỉnh có biển báo/đèn. Vùng xuất phát và vùng kết thúc đánh dấu rõ.

### Loại tín hiệu

| Loại | Yêu cầu xe |
|---|---|
| **Biển báo lệnh** | Bắt buộc rời giao lộ theo hướng X ∈ {thẳng, trái, phải} |
| **Biển báo cấm** | Cấm rời theo hướng bị cấm → chọn hướng hợp lệ khác (**chỉ có 2 hướng còn lại**) |
| **Đèn xanh** | Được vượt vạch dừng, đi tiếp |
| **Đèn đỏ** | **Phải dừng trước vạch dừng** đến khi chuyển xanh |

> ⚠ **Trái/phải/thẳng tính theo hướng di chuyển hiện tại của xe khi tiến vào giao lộ**, KHÔNG theo hướng tuyệt đối của bản đồ. → FSM phải làm việc trong hệ toạ độ thân xe, không cần la bàn.

### Nhiệm vụ

- [ ] **Nhận diện biển báo** — nhãn đúng + hành động đúng tại giao lộ tương ứng
- [ ] **Ra quyết định tại giao lộ** — rời đúng hướng yêu cầu / tránh hướng cấm
- [ ] **Tuân thủ đèn** — dừng trước vạch khi đỏ, chỉ đi khi được phép
- [ ] **Đến vùng kết thúc** — **toàn bộ thân xe** vào vùng kết thúc trong 5 phút
- [ ] **Ghi log xử lý** — chứng minh được **latency ≤ 300 ms** từ lúc biển hiện rõ đến khi sinh lệnh

### Điểm

```
Điểm lượt = max(0, Biển báo + Time − Penalty)
Biển báo = (số biển đọc được / tổng số biển) × 70, max 70   ⚠ công thức gốc viết sai, xem BASELINE §6.2
Time     = 30 − 1 × (t/10), không âm
Điểm cuối = MAX của 2 lượt
```

### Lỗi

| Vi phạm | Kết quả |
|---|---|
| **Vượt đèn đỏ** | **HỦY LƯỢT** |
| Không dừng trước vạch khi đèn đỏ | −10/lần |
| **Can thiệp thủ công** | **HỦY LƯỢT** |
| Đi sai lộ trình | **Kết thúc lượt**, tính điểm tới biển đúng xa vạch xuất phát nhất |

---

## D. Xử lý sự cố (ĐB §6)

| Tình huống | Điều kiện xác định | Xử lý |
|---|---|---|
| Lệch đường đua | 2 bánh cùng bên ra khỏi mép | BTC đưa về checkpoint gần nhất + phạt |
| Xe kẹt/dừng | Đứng yên **10 giây** | Đội được yêu cầu restart từ checkpoint/giao lộ gần nhất. **Đồng hồ KHÔNG dừng** (trừ lỗi thiết bị BTC) |
| Lỗi thiết bị BTC | Không do đội gây ra | BTC xem xét chạy lại / tạm dừng đồng hồ |
| Hết giờ lượt | Đủ 5 phút | Dừng ngay, tính đến checkpoint/giao lộ cuối hợp lệ |
| Tranh chấp | Không đồng ý kết quả | Gửi trong **5 phút** sau công bố điểm; BTC xét log/camera/biên bản |

> Hệ quả kỹ thuật: xe kẹt 10 giây mà đồng hồ vẫn chạy → cần **watchdog tự phát hiện kẹt** (vận tốc ≈ 0 và cte không đổi trong ~3 s) và tự thoát (lùi nhẹ + đánh lái) *trước khi* chạm mốc 10 giây phải xin restart.

---

## E. Xếp hạng & tie-break

Tổng = ST×30% + SC×40% + Paper×20% + Defense×10%.

Tie-break (ĐB §5 / Thể lệ §10.2): **Smart City → Speed Track → (Paper → Defense) → tổng thời gian thấp hơn → ít lỗi hơn → BTC quyết định.**

---

## F. Technical Paper (20%) — mục bắt buộc

Độ dài **6–8 trang**, IEEE rút gọn. Mục bắt buộc:

- [ ] Introduction
- [ ] Related Work
- [ ] Method
- [ ] Experimental Setup
- [ ] Results
- [ ] Discussion
- [ ] Limitation
- [ ] References

Trọng tâm chấm (Thể lệ §8.2): **phân tích định lượng qua log** — bảng số liệu, metric, biểu đồ; **lập luận lựa chọn và so sánh mô hình** (trade-off); không mô tả cảm tính. Khuyến khích ablation.

Ràng buộc học thuật (Thể lệ §9): tài liệu tham khảo **phải có thật, kiểm chứng được**; không số liệu giả; được dùng AI hỗ trợ nhưng **phải giải thích được mọi thứ đã nộp khi bị hỏi**.

---

## G. Oral Defense (10%)

Trình bày: vấn đề → phương pháp → kết quả → minh chứng → bài học. Chuẩn bị trả lời được: *"vì sao chọn phương pháp này thay vì X?"*, *"con số này lấy từ đâu?"*, *"hệ thống hỏng trong trường hợp nào?"*
