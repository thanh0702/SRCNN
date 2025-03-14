# SRCNN

## 📥 Download Dataset and test
- *Dataset*: [Download here](https://drive.google.com/file/d/1WUHVyr9ciQvitIe50UpCixUQB7xK43pQ/view?usp=sharing)
- *test*: [Download here](https://drive.google.com/file/d/1HcaxxUXmyRsLcEqX5LL_TiiV9_2vvzm5/view?usp=sharing)

## 📂 Cấu trúc thư mục
```

SRCNN/
│── output/                            # Thư mục chứa dữ liệu hình ảnh đầu ra
│── dataset/                           # Thư mục chứa dữ liệu huấn luyện mô hình
│   │── train/                         # Thư mục huấn luyện mô hình
│       │── original                   # Thư mục chứa ảnh HR pic để huấn luyện
│           │── HR pic                 # Thư mục chứa dữ liệu hình ảnh cao (HR pic)
│       │── DIV2K_train_LR_bicubic/    # Thư mục thực nghiệm mô hình
│           │── x2/                    # Thư mục chứa ảnh LR pic để thực nghiệm
│               │── LR pic             # Thư mục chứa dữ liệu hình ảnh thấp (LR pic)
│── test/                              # Thư mục chứa dữ liệu thực nghiệm mô hình       
│   │── validation-HR/                 # Thư mục chứa ảnh HR pic để thực nghiệm
│       │── HR pic                     # Thư mục chứa dữ liệu hình ảnh cao (HR pic)
│   │── validation-LRX2/               # Thư mục chứa ảnh LR pic để thực nghiệm
│       │── LR pic                     # Thư mục chứa dữ liệu hình ảnh thấp (LR pic)
│   │── validation-LRX3/               # Thư mục chứa ảnh LR pic để thực nghiệm
│       │── LR pic                     # Thư mục chứa dữ liệu hình ảnh thấp (LR pic)
│   │── validation-LRX4/               # Thư mục chứa ảnh LR pic để thực nghiệm
│       │── LR pic                     # Thư mục chứa dữ liệu hình ảnh thấp (LR pic)
│   │── validation-LRX8/               # Thư mục chứa ảnh LR pic để thực nghiệm
│       │── LR pic                     # Thư mục chứa dữ liệu hình ảnh thấp (LR pic)
│── train_srcnn.py                     # Script huấn luyện mô hình
│── inference_srcnn.py                 # Script chạy mô hình để suy luận
│── srcnn.pth                          # Tệp mô hình đã huấn luyện
│── 3DoDoAnh.py                        # Script đo Tính Entropy, phương sai, độ tương phản ảnh
│── ssim.py                            # Script đo SSIM ảnh
│── psnr.py                            # Script đo PSNR ảnh màu và ảnh xám
│── README.md                          # Hướng dẫn sử dụng
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
