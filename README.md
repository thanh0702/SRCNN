# SRCNN

## 📥 Download Dataset & Data
- *Dataset*: [Download here](https://drive.google.com/file/d/1WUHVyr9ciQvitIe50UpCixUQB7xK43pQ/view?usp=sharing)
- *Data*: [Download here](https://drive.google.com/file/d/1UeB6IZ04xK2N524JLi91FapzEGMhTP90/view?usp=sharing)

## 📂 Cấu trúc thư mục
```

SRCNN/
│── data/                   # Thư mục chứa dữ liệu hình ảnh thấp (LR pic)
│── dataset/                # Thư mục chứa dữ liệu huấn luyện mô hình
│   │── train/              # Thư mục huấn luyện mô hình
│       │── original        # Thư mục chứa ảnh HR pic để huấn luyện
│           │── HR pic      # Thư mục chứa dữ liệu hình ảnh cao (HR pic)
│── train_srcnn.py          # Script huấn luyện mô hình
│── inference_srcnn.py      # Script chạy mô hình để suy luận
│── srcnn.pth               # Tệp mô hình đã huấn luyện
│── ssim.py                 # Script đo SSIM ảnh màu
│── grayssim.py             # Script đo SSIM ảnh xám
│── bicubic.py              # Script giảm độ phân giải ảnh bằng Bicubic
│── psnr.py                 # Script đo PSNR ảnh màu và ảnh xám
│── README.md               # Hướng dẫn sử dụng
```

## 📜 Mô tả
SRCNN (Super-Resolution Convolutional Neural Network) là một mô hình học sâu sử dụng mạng CNN để nâng cấp chất lượng hình ảnh từ độ phân giải thấp (LR) lên độ phân giải cao (HR). 

📌 *Các thành phần chính trong dự án:*
- **train_srcnn.py**: Dùng để huấn luyện mô hình SRCNN.
- **inference_srcnn.py**: Dùng để thực hiện suy luận trên ảnh đầu vào.
- **ssim.py & grayssim.py**: Đo độ tương đồng SSIM giữa ảnh gốc và ảnh tái tạo.
- **bicubic.py**: Giảm độ phân giải ảnh bằng thuật toán nội suy Bicubic.
- **psnr.py**: Đánh giá chất lượng ảnh bằng chỉ số PSNR.

🔗 *Hướng dẫn sử dụng chi tiết* có thể tham khảo trong README.md. 🚀
