import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image

# -------------------------------
# 1⃣ Định nghĩa mô hình SRCNN
# -------------------------------
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

# -------------------------------
# 2⃣ Load mô hình đã huấn luyện
# -------------------------------
def load_model(model_path):
    model = SRCNN(num_channels=3)  # 3 kênh (RGB)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# -------------------------------
# 3⃣ Tiền xử lý ảnh trước khi đưa vào mô hình
# -------------------------------
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Thêm batch dimension

# -------------------------------
# 4⃣ Chạy mô hình & lưu kết quả (RGB)
# -------------------------------
def save_output(tensor, output_path):
    image = tensor.squeeze(0).detach().cpu().clamp(0, 1)  # Giới hạn giá trị từ [0,1]
    image = transforms.ToPILImage()(image)  # Chuyển tensor thành ảnh RGB
    image.save(output_path)  # Lưu ảnh

# -------------------------------
# 5⃣ Chạy chương trình Inference
# -------------------------------
def run_inference(image_path, model_path, output_path):
    model = load_model(model_path)
    input_image = preprocess_image(image_path)
    
    # Chạy mô hình
    with torch.no_grad():
        output_image = model(input_image)

    # Lưu ảnh kết quả
    save_output(output_image, output_path)
    print(f"✅ Ảnh đã được xử lý và lưu tại: {output_path}")

# -------------------------------
# 6⃣ Chạy chương trình
# -------------------------------
if __name__ == "__main__":
    image_path = "D:/srcnn/data/chim.png"  # Đường dẫn ảnh gốc
    model_path = "D:/srcnn/srcnn.pth"  # Đường dẫn mô hình đã huấn luyện
    output_path = "D:/srcnn/data/output.jpg"  # Lưu ảnh kết quả
    
    run_inference(image_path, model_path, output_path)
