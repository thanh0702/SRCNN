import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

def select_image(title="Chọn ảnh"):
    """Hàm chọn ảnh từ file"""
    root = Tk()
    root.withdraw()  # Ẩn cửa sổ chính
    image_path = filedialog.askopenfilename(title=title, filetypes=[("Ảnh", "*.png;*.jpg;*.jpeg;*.bmp")])
    if not image_path:
        print("❌ Không có ảnh nào được chọn!")
        return None
    return image_path

def calculate_psnr(img1, img2):
    """Tính PSNR giữa hai ảnh"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # Ảnh giống hệt nhau
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

# 📌 Chọn ảnh gốc và ảnh đã xử lý
original_path = select_image("Chọn ảnh gốc")
if not original_path:
    exit()
processed_path = select_image("Chọn ảnh đã xử lý")
if not processed_path:
    exit()

# 📌 Đọc ảnh
original = cv2.imread(original_path)
processed = cv2.imread(processed_path)

# Kiểm tra ảnh có tồn tại không
if original is None or processed is None:
    raise ValueError("❌ Không thể đọc một trong hai ảnh!")

# 📌 Chuyển từ BGR sang RGB để hiển thị đúng màu
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

# 🔹 Tính PSNR trên toàn bộ ảnh màu (RGB)
psnr_rgb = calculate_psnr(original, processed)

# 🔹 Chuyển đổi ảnh sang ảnh xám (Grayscale)
original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

# 🔹 Tính PSNR trên ảnh xám
psnr_gray = calculate_psnr(original_gray, processed_gray)

# 🔹 Tính PSNR riêng cho từng kênh màu
psnr_red = calculate_psnr(original[:, :, 2], processed[:, :, 2])  # Kênh đỏ
psnr_green = calculate_psnr(original[:, :, 1], processed[:, :, 1])  # Kênh xanh lá
psnr_blue = calculate_psnr(original[:, :, 0], processed[:, :, 0])  # Kênh xanh dương

# 🔹 Tính PSNR trung bình từ ba kênh màu
psnr_mean = (psnr_red + psnr_green + psnr_blue) / 3

# 📌 Hiển thị ảnh gốc và ảnh đã xử lý
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_rgb)
plt.title("Ảnh Gốc")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(processed_rgb)
plt.title("Ảnh Đã Xử Lý")
plt.axis('off')

plt.show()

# 📌 Hiển thị kết quả PSNR
print(f"🔹 PSNR trên ảnh màu (RGB): {psnr_rgb:.2f} dB")
print(f"🔹 PSNR kênh Đỏ (Red): {psnr_red:.2f} dB")
print(f"🔹 PSNR kênh Xanh Lá (Green): {psnr_green:.2f} dB")
print(f"🔹 PSNR kênh Xanh Dương (Blue): {psnr_blue:.2f} dB")
print(f"🔹 PSNR Trung Bình 3 Kênh Màu: {psnr_mean:.2f} dB")
print(f"🔹 PSNR trên ảnh xám (Grayscale): {psnr_gray:.2f} dB")
