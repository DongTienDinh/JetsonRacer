import traitlets
import ipywidgets.widgets as widgets
from jetracer.nvidia_racecar import NvidiaRacecar

# ========================================================
# 1. KHỞI TẠO XE & MỞ KHÓA GÓC CUA GẮT
# ========================================================
car = NvidiaRacecar()

try:
    car.steering_motor.set_pulse_width_range(500, 2500)
    print("✅ Đã mở khóa giới hạn góc cua vật lý (PWM: 500 - 2500) thành công!")
except Exception as e:
    print("⚠️ Không thể can thiệp xung PWM:", e)

car.steering_gain = -1.0


# ========================================================
# 2. KHỞI TẠO GIAO DIỆN TAY CẦM
# ========================================================
controller = widgets.Controller(index=0)
display(controller)
print("👉 CHÚ Ý: Hãy bấm một nút bất kỳ hoặc xoay nhẹ 2 cần gạt để kích hoạt tay cầm.")


# ========================================================
# 3. NÚT BẤM KẾT NỐI (ĐÃ TÁCH 2 CẦN GẠT TRÁI/PHẢI)
# ========================================================
btn_connect = widgets.Button(description="KÍCH HOẠT ĐIỀU KHIỂN XE", button_style='success', layout=widgets.Layout(width='300px', height='40px'))
output = widgets.Output()

def thuc_hien_ket_noi(b):
    with output:
        output.clear_output()
        # Đảm bảo tay cầm đã nhận diện đủ trục (ít nhất 3 trục cho 2 cần gạt)
        if len(controller.axes) >= 3:
            try:
                # ĐÃ ĐỔI THÀNH AXES[2]: Dùng Cần gạt PHẢI để bẻ lái (Trái/Phải)
                traitlets.dlink((controller.axes[2], 'value'), (car, 'steering'), transform=lambda x: x)
                
                # GIỮ NGUYÊN AXES[1]: Dùng Cần gạt TRÁI để chạy (Tiến/Lùi)
                traitlets.dlink((controller.axes[1], 'value'), (car, 'throttle'), transform=lambda x: -x)
                
                print("🚀 [THÀNH CÔNG] Đã tách cần gạt thành công!")
                print("🕹️ Cần TRÁI: Tiến / Lùi")
                print("🕹️ Cần PHẢI: Bẻ lái Trái / Phải")
            except Exception as errors:
                print(f"❌ Lỗi khi thiết lập liên kết: {errors}")
        else:
            print("❌ [THẤT BẠI] Trình duyệt chưa nhận diện đủ cần gạt.")
            print("👉 Sửa lỗi: Bạn hãy xoay tròn cả 2 cần gạt trên tay cầm vài vòng, sau đó bấm lại nút này.")

btn_connect.on_click(thuc_hien_ket_noi)

print("-" * 60)
display(btn_connect, output)