from skimage.metrics import structural_similarity as ssim
import cv2

# Đọc ảnh gốc và ảnh đã xử lý
original_image = cv2.imread("D:/srcnn/data/vit.jpg")
processed_image = cv2.imread("D:/srcnn/data/output.jpg")

# Kiểm tra xem ảnh có được đọc thành công không
if original_image is None:
    raise ValueError("Không thể đọc ảnh gốc từ đường dẫn: duong_dan_den_anh_goc.jpg")
if processed_image is None:
    raise ValueError("Không thể đọc ảnh đã xử lý từ đường dẫn: duong_dan_den_anh_xu_ly.jpg")

# Chuyển đổi ảnh sang định dạng RGB
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

# Tính toán SSIM cho từng kênh màu
ssim_red = ssim(original_image[:, :, 0], processed_image[:, :, 0], data_range=255)
ssim_green = ssim(original_image[:, :, 1], processed_image[:, :, 1], data_range=255)
ssim_blue = ssim(original_image[:, :, 2], processed_image[:, :, 2], data_range=255)

# Chuyển đổi ảnh sang thang độ xám
original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
processed_gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

# Tính toán SSIM giữa hai ảnh
ssim_value = ssim(original_gray, processed_gray)

print(f'Chỉ số SSIM giữa hai ảnh xám: {ssim_value}')

# Tính trung bình SSIM cho cả ba kênh màu
ssim_mean = (ssim_red + ssim_green + ssim_blue) / 3

print(f'Chỉ số SSIM cho kênh đỏ: {ssim_red}')
print(f'Chỉ số SSIM cho kênh lục: {ssim_green}')
print(f'Chỉ số SSIM cho kênh lam: {ssim_blue}')
print(f'Chỉ số SSIM trung bình: {ssim_mean}')
