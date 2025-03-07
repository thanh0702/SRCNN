from skimage.metrics import structural_similarity as ssim
import cv2

# Đọc ảnh gốc và ảnh đã xử lý
original_image = cv2.imread("D:/srcnn/data/sutu.jpg")
processed_image = cv2.imread("D:/srcnn/data/output.jpg")

# Kiểm tra xem ảnh có được đọc thành công không
if original_image is None:
    raise ValueError("Không thể đọc ảnh gốc từ đường dẫn: duong_dan_den_anh_goc.jpg")
if processed_image is None:
    raise ValueError("Không thể đọc ảnh đã xử lý từ đường dẫn: duong_dan_den_anh_xu_ly.jpg")

# Chuyển đổi ảnh sang thang độ xám
original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
processed_gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

# Tính toán SSIM giữa hai ảnh
ssim_value = ssim(original_gray, processed_gray)

print(f'Chỉ số SSIM giữa hai ảnh: {ssim_value}')
