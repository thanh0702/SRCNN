import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from PIL import Image

# -------------------------------
# 1️⃣ Định nghĩa mô hình SRCNN
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
# 2️⃣ Chuẩn bị dữ liệu
# -------------------------------
def get_dataloader(data_path, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize ảnh về 128x128
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

# -------------------------------
# 3️⃣ Huấn luyện mô hình
# -------------------------------
def train(model, dataloader, num_epochs=10, lr=0.001, save_path="srcnn.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()  # Hàm mất mát
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Optimizer
    
    loss_history = []  # Lưu lịch sử loss

    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, (inputs, _) in progress_bar:
            inputs = inputs.to(device)
            
            # Tạo ảnh độ phân giải thấp bằng cách downscale rồi upscale
            lr_images = transforms.Resize((64, 64))(inputs)  # Giảm kích thước xuống 64x64
            lr_images = transforms.Resize((128, 128))(lr_images)  # Phóng to lại 128x128

            optimizer.zero_grad()
            outputs = model(lr_images)

            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(dataloader)
        loss_history.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Lưu mô hình
    torch.save(model.state_dict(), save_path)
    print(f"✅ Mô hình đã được lưu tại: {save_path}")

    # Vẽ biểu đồ loss
    plt.plot(range(1, num_epochs + 1), loss_history, marker='o', label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss during training")
    plt.savefig("training_loss.png")
    plt.show()

# -------------------------------
# 4️⃣ Chạy chương trình huấn luyện
# -------------------------------
if __name__ == "__main__":
    data_path = "D:/srcnn/dataset/train"  # Thư mục chứa dữ liệu
    model_save_path = "D:/srcnn/srcnn.pth"  # Lưu mô hình

    batch_size = 16
    num_epochs = 20
    learning_rate = 0.001

    model = SRCNN(num_channels=3)
    dataloader = get_dataloader(data_path, batch_size)

    train(model, dataloader, num_epochs, learning_rate, model_save_path)
