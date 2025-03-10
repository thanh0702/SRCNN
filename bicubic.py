import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh HR
image_hr = cv2.imread("D:/srcnn/data/vit.jpg")
image_hr = cv2.cvtColor(image_hr, cv2.COLOR_BGR2RGB)  # Chuyển từ BGR sang RGB

# Xác định tỷ lệ downscale (ví dụ: giảm kích thước xuống 4 lần)
scale_factor = 2
height, width = image_hr.shape[:2]
new_size = (width // scale_factor, height // scale_factor)

# Tạo ảnh LR bằng Bicubic Interpolation
image_lr = cv2.resize(image_hr, new_size, interpolation=cv2.INTER_CUBIC)

# Hiển thị ảnh
plt.figure(figsize=(10,5))
plt.subplot(1,2,1), plt.imshow(image_hr), plt.title("Ảnh HR (Gốc)"), plt.axis("off")
plt.subplot(1,2,2), plt.imshow(image_lr), plt.title("Ảnh LR (Bicubic)"), plt.axis("off")
plt.show()

# Lưu ảnh LR
cv2.imwrite("D:/srcnn/data/low_res.jpg", cv2.cvtColor(image_lr, cv2.COLOR_RGB2BGR))
