import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import filedialog

# 1️⃣ Định nghĩa mô hình SRCNN 
class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, stride=1, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, stride=1, padding=2)
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

# 2️⃣ Load mô hình đã huấn luyện
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SRCNN().to(DEVICE)
model.load_state_dict(torch.load("D:/srcnn/srcnn.pth", map_location=DEVICE))
model.eval()

# 3️⃣ Hàm xử lý ảnh đầu vào và chạy mô hình
def enhance_image(image_path):
    # Load ảnh
    img = Image.open(image_path).convert("RGB")

    # Tăng kích thước ảnh 
    img_lr = img.resize((img.width * 2, img.height * 2), Image.BICUBIC)

    # Chuyển ảnh sang tensor
    transform = transforms.ToTensor()
    img_lr_tensor = transform(img_lr).unsqueeze(0).to(DEVICE)

    # Chạy mô hình SRCNN
    with torch.no_grad():
        img_sr_tensor = model(img_lr_tensor)

    # Chuyển tensor về ảnh
    img_sr = img_sr_tensor.squeeze(0).cpu().numpy()
    img_sr = np.clip(img_sr.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    img_sr = Image.fromarray(img_sr)

    # Lưu ảnh kết quả tại đường dẫn cố định
    output_path = "D:/srcnn/output/output.png"
    img_sr.save(output_path)
    print(f"✅ Ảnh đầu ra đã được lưu tại: {output_path}")

# 4️⃣ Chọn ảnh từ hộp thoại
def select_and_process_image():
    root = tk.Tk()
    root.withdraw()  # Ẩn cửa sổ chính

    # Chọn ảnh đầu vào
    image_input = filedialog.askopenfilename(title="Chọn ảnh cần xử lý", filetypes=[("PNG files", "*.png"), ("JPG files", "*.jpg"), ("All files", "*.*")])
    
    if not image_input:
        print("❌ Không có ảnh nào được chọn.")
        return

    # Xử lý ảnh
    enhance_image(image_input)

# 5️⃣ Chạy chương trình
select_and_process_image()
