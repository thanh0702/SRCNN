from skimage.metrics import structural_similarity as ssim
import cv2
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

# 📌 Chọn ảnh gốc và ảnh đã xử lý
original_path = select_image("Chọn ảnh gốc")
if not original_path:
    exit()
processed_path = select_image("Chọn ảnh đã xử lý")
if not processed_path:
    exit()

# 📌 Đọc ảnh
original_image = cv2.imread(original_path)
processed_image = cv2.imread(processed_path)

# Kiểm tra xem ảnh có được đọc thành công không
if original_image is None:
    raise ValueError(f"❌ Không thể đọc ảnh gốc từ đường dẫn: {original_path}")
if processed_image is None:
    raise ValueError(f"❌ Không thể đọc ảnh đã xử lý từ đường dẫn: {processed_path}")

# 📌 Chuyển đổi ảnh sang định dạng RGB
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

# 🔹 Tính toán SSIM cho từng kênh màu
ssim_red = ssim(original_image[:, :, 0], processed_image[:, :, 0], data_range=255)
ssim_green = ssim(original_image[:, :, 1], processed_image[:, :, 1], data_range=255)
ssim_blue = ssim(original_image[:, :, 2], processed_image[:, :, 2], data_range=255)

# 🔹 Chuyển đổi ảnh sang thang độ xám
original_gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
processed_gray = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)

# 🔹 Tính toán SSIM giữa hai ảnh xám
ssim_value = ssim(original_gray, processed_gray, data_range=255)

# 🔹 Tính trung bình SSIM cho cả ba kênh màu
ssim_mean = (ssim_red + ssim_green + ssim_blue) / 3

# 🔹 In kết quả
print(f"📊 Chỉ số SSIM giữa hai ảnh xám: {ssim_value:.4f}")
print(f"🔴 Chỉ số SSIM cho kênh đỏ: {ssim_red:.4f}")
print(f"🟢 Chỉ số SSIM cho kênh lục: {ssim_green:.4f}")
print(f"🔵 Chỉ số SSIM cho kênh lam: {ssim_blue:.4f}")
print(f"📈 Chỉ số SSIM trung bình: {ssim_mean:.4f}")
