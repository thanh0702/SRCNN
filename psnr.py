import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_psnr(img1, img2):
    """Tính PSNR giữa hai ảnh"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # Ảnh giống hệt nhau
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

# Đọc ảnh gốc và ảnh đã xử lý
original = cv2.imread('D:/srcnn/data/vit.jpg')
processed = cv2.imread('D:/srcnn/data/output.jpg')

# Chuyển từ BGR sang RGB để hiển thị đúng màu
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

# 1️⃣ Tính PSNR trên toàn bộ ảnh màu (RGB)
psnr_rgb = calculate_psnr(original, processed)

# 2️⃣ Chuyển đổi ảnh sang ảnh xám (Grayscale)
original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

# 3️⃣ Tính PSNR trên ảnh xám
psnr_gray = calculate_psnr(original_gray, processed_gray)

# Hiển thị kết quả
plt.figure(figsize=(10, 5))
plt.subplot(1,2,1), plt.imshow(original_rgb), plt.title("Ảnh Gốc"), plt.axis('off')
plt.subplot(1,2,2), plt.imshow(processed_rgb), plt.title("Ảnh Đã Xử Lý"), plt.axis('off')
plt.show()

print(f"🔹 PSNR trên ảnh màu (RGB): {psnr_rgb:.2f} dB")
print(f"🔹 PSNR trên ảnh xám (Grayscale): {psnr_gray:.2f} dB")
