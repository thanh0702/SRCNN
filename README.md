# SRCNN
download dataset https://drive.google.com/file/d/1WUHVyr9ciQvitIe50UpCixUQB7xK43pQ/view?usp=sharing
download data https://drive.google.com/file/d/1UeB6IZ04xK2N524JLi91FapzEGMhTP90/view?usp=sharing
Mô hình 
SRCNN/
│── data/                   # Thư mục chứa dữ liệu hình ảnh thấp LR pic
│── dataset/                # Thư mục chứa huấn luyện mô hình
│   │── train/              # Thư mục huấn luyện mô hình
│       │── original        # Thư mục chứa ảnh HR pic để huấn luyện mô hình
│           │── HR pic      # Thư mục chứa dữ liệu hình ảnh cao HR pic
│── train_srcnn.py          # Tập lệnh huấn luyện mô hình
│── inference_srcnn.py      # Tập lệnh sử dụng mô hình
│── srcnn.pth               # Tệp mô hình đã huấn luyện
│── ssim.py                 # Tập lệnh đo ssim ảnh màu
│── grayssim.py             # Tập lệnh đo ssim ảnh xám
│── bicubic.py              # Tập lệnh giảm độ phân giải ảnh bicubic
│── psnr.py                 # Tập lệnh đo psnr ảnh màu và ảnh xám
│── README.md               # Hướng dẫn sử dụng 

