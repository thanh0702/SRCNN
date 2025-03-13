import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from tkinter import Tk, filedialog

def calculate_entropy(image):
    """Tính Entropy của một ảnh"""
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    hist = hist / hist.sum()  # Chuẩn hóa histogram thành xác suất
    return entropy(hist, base=2)  # Tính entropy dựa trên xác suất pixel

def calculate_variance(image):
    """Tính phương sai (độ sắc nét) của ảnh"""
    return np.var(image)

def calculate_contrast(image):
    """Tính độ tương phản RMS (Root Mean Square Contrast)"""
    return np.sqrt(np.mean(np.square(image - np.mean(image))))

def select_image(title="Chọn ảnh"):
    """Hàm chọn ảnh từ file"""
    root = Tk()
    root.withdraw()  # Ẩn cửa sổ chính
    image_path = filedialog.askopenfilename(title=title, filetypes=[("Ảnh", "*.png;*.jpg;*.jpeg;*.bmp")])
    if not image_path:
        print("❌ Không có ảnh nào được chọn!")
        return None
    return image_path

def process_image(image_path):
    """Đọc ảnh và tính các chỉ số"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ Không thể đọc ảnh từ đường dẫn: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB

    # Tính các chỉ số cho từng kênh màu
    entropy_vals = [calculate_entropy(image_rgb[:, :, i]) for i in range(3)]
    variance_vals = [calculate_variance(image_rgb[:, :, i]) for i in range(3)]
    contrast_vals = [calculate_contrast(image_rgb[:, :, i]) for i in range(3)]

    # Tính trung bình
    entropy_mean = np.mean(entropy_vals)
    variance_mean = np.mean(variance_vals)
    contrast_mean = np.mean(contrast_vals)

    return image_rgb, entropy_vals, variance_vals, contrast_vals, entropy_mean, variance_mean, contrast_mean

# 📌 Chọn hai ảnh
image_path1 = select_image("Chọn ảnh thứ nhất")
if not image_path1:
    exit()
image_path2 = select_image("Chọn ảnh thứ hai")
if not image_path2:
    exit()

# 🔹 Xử lý cả hai ảnh
image1, entropy1, variance1, contrast1, entropy_mean1, variance_mean1, contrast_mean1 = process_image(image_path1)
image2, entropy2, variance2, contrast2, entropy_mean2, variance_mean2, contrast_mean2 = process_image(image_path2)

# 🔹 Hiển thị ảnh và kết quả riêng biệt
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image1)
axes[0].set_title("Ảnh 1")
axes[0].axis("off")
axes[1].imshow(image2)
axes[1].set_title("Ảnh 2")
axes[1].axis("off")
plt.show()

# 🔹 Hiển thị kết quả của từng ảnh
print("📌 KẾT QUẢ ẢNH 1:")
print(f"📊 Entropy (R,G,B): {entropy1[0]:.4f}, {entropy1[1]:.4f}, {entropy1[2]:.4f} | Trung bình: {entropy_mean1:.4f}")
print(f"🔍 Phương sai (R,G,B): {variance1[0]:.4f}, {variance1[1]:.4f}, {variance1[2]:.4f} | Trung bình: {variance_mean1:.4f}")
print(f"🌟 Độ tương phản (R,G,B): {contrast1[0]:.4f}, {contrast1[1]:.4f}, {contrast1[2]:.4f} | Trung bình: {contrast_mean1:.4f}")

print("\n📌 KẾT QUẢ ẢNH 2:")
print(f"📊 Entropy (R,G,B): {entropy2[0]:.4f}, {entropy2[1]:.4f}, {entropy2[2]:.4f} | Trung bình: {entropy_mean2:.4f}")
print(f"🔍 Phương sai (R,G,B): {variance2[0]:.4f}, {variance2[1]:.4f}, {variance2[2]:.4f} | Trung bình: {variance_mean2:.4f}")
print(f"🌟 Độ tương phản (R,G,B): {contrast2[0]:.4f}, {contrast2[1]:.4f}, {contrast2[2]:.4f} | Trung bình: {contrast_mean2:.4f}")

# 🔹 So sánh và in chênh lệch
print("\n📊 SO SÁNH CHÊNH LỆCH GIỮA ẢNH 1 VÀ ẢNH 2:")
print(f"📊 Entropy trung bình chênh lệch: {abs(entropy_mean1 - entropy_mean2):.4f}")
print(f"🔍 Phương sai trung bình chênh lệch: {abs(variance_mean1 - variance_mean2):.4f}")
print(f"🌟 Độ tương phản trung bình chênh lệch: {abs(contrast_mean1 - contrast_mean2):.4f}")
