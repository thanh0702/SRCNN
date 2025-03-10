import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# 1️⃣ Định nghĩa mô hình SRCNN (giống như khi huấn luyện)
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
def enhance_image(image_path, output_path):
    # Load ảnh
    img = Image.open(image_path).convert("RGB")

    # Tăng kích thước ảnh để phù hợp với mô hình
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

    # Lưu ảnh kết quả
    img_sr.save(output_path)
    print(f"✅ Ảnh đầu ra đã được lưu tại: {output_path}")

# 4️⃣ Sử dụng mô hình trên ảnh mẫu
image_input = "D:/srcnn/data/low_res.jpg"  # Đường dẫn ảnh đầu vào
image_output = "D:/srcnn/data/output.jpg"  # Đường dẫn lưu ảnh đầu ra
enhance_image(image_input, image_output)
