import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

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
    def __init__(self, root_dir, transform=None, image_size=(256, 256)):  # ✅ Thêm tham số image_size
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size  # ✅ Lưu kích thước ảnh chuẩn
        self.image_filenames = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".png")]
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_filenames[idx]).convert("RGB")
        
        # ✅ Resize ảnh về cùng kích thước
        img = img.resize(self.image_size, Image.BICUBIC)
        
        img_hr = self.transform(img)
        img_lr = img.resize((self.image_size[0] // 2, self.image_size[1] // 2), Image.BICUBIC)
        img_lr = img_lr.resize(self.image_size, Image.BICUBIC)
        img_lr = self.transform(img_lr)
        
        return img_lr, img_hr

# 3️⃣ Cấu hình huấn luyện
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
IMAGE_SIZE = (256, 256)  # ✅ Chọn kích thước ảnh cố định

transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = SRDataset(root_dir="D:/srcnn/dataset/train/original", transform=transform, image_size=IMAGE_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 4️⃣ Khởi tạo mô hình và tối ưu hóa
model = SRCNN().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 5️⃣ Vòng lặp huấn luyện
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
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss/len(dataloader):.6f}")

# 6️⃣ Lưu mô hình
torch.save(model.state_dict(), "D:/srcnn/srcnn.pth")
print("✅ Huấn luyện hoàn tất, mô hình đã được lưu!")
