import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import matplotlib.pyplot as plt

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

# 2️⃣ Tiền xử lý dữ liệu
class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, transform=None, image_size=(256, 256)):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.transform = transform
        self.image_size = image_size
        self.image_filenames = [f for f in os.listdir(hr_dir) if f.endswith(".png")]
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        hr_path = os.path.join(self.hr_dir, self.image_filenames[idx])
        lr_path = os.path.join(self.lr_dir, self.image_filenames[idx].replace(".png", "x2.png"))
        
        if not os.path.exists(lr_path):
            raise FileNotFoundError(f"Không tìm thấy ảnh LR: {lr_path}")
        
        img_hr = Image.open(hr_path).convert("RGB").resize(self.image_size, Image.BICUBIC)
        img_lr = Image.open(lr_path).convert("RGB").resize(self.image_size, Image.BICUBIC)
        
        img_hr = self.transform(img_hr)
        img_lr = self.transform(img_lr)

        return img_lr, img_hr

# 3️⃣ Cấu hình huấn luyện
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
IMAGE_SIZE = (256, 256)

transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = SRDataset(
    hr_dir="D:/srcnn/dataset/train/original",
    lr_dir="D:/srcnn/dataset/train/DIV2K_train_LR_bicubic/X2",
    transform=transform,
    image_size=IMAGE_SIZE
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 4️⃣ Khởi tạo mô hình và tối ưu hóa
model = SRCNN().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 5️⃣ Vòng lặp huấn luyện + Lưu lịch sử loss
loss_history = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for lr_imgs, hr_imgs in dataloader:
        lr_imgs, hr_imgs = lr_imgs.to(DEVICE), hr_imgs.to(DEVICE)
        optimizer.zero_grad()
        sr_imgs = model(lr_imgs)
        loss = criterion(sr_imgs, hr_imgs)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    loss_history.append(avg_loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")

# 6️⃣ Lưu mô hình
model_path = "D:/srcnn/srcnn.pth"
torch.save(model.state_dict(), model_path)
print(f"✅ Huấn luyện hoàn tất, mô hình đã được lưu tại: {model_path}")

# 7️⃣ Vẽ biểu đồ loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS + 1), loss_history, marker='o', linestyle='-')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid(True)

# Lưu hình ảnh loss vào D:/srcnn
loss_plot_path = "D:/srcnn/training_loss.png"
plt.savefig(loss_plot_path)
plt.show()
print(f"✅ Biểu đồ loss đã được lưu tại: {loss_plot_path}")
