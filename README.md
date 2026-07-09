# VA-π Mini — Cài đặt hệ thống sinh ảnh tự hồi quy (Autoregressive Generation) bằng GRPO

Dự án này là phiên bản thu nhỏ (Mini implementation) của framework **VA-π** (Value-Aligned Autoregressive Generation). Mục tiêu cốt lõi của dự án là hiện thực hóa và kiểm thử thuật toán căn chỉnh học tăng cường (RL Alignment) sử dụng **Group Relative Policy Optimization (GRPO)** trên môi trường hạn chế tài nguyên phần cứng (như GPU miễn phí của Kaggle).

Dự án triển khai đầy đủ cả 3 giai đoạn của quy trình nghiên cứu thực nghiệm:
1. **Giai đoạn 1**: Huấn luyện từ đầu (Train from scratch) tokenizer VQ-VAE và mô hình tự hồi quy GPT-Mini trên tập dữ liệu **CIFAR-10** để làm checkpoint nền tảng.
2. **Giai đoạn 2**: Đánh giá chất lượng cơ sở (Sanity Check) của checkpoint nền tảng bằng các chỉ số Reconstruction Error, FID, IS, Precision và Recall.
3. **Giai đoạn 3 (Evaluation)**: Áp dụng thuật toán **VA-π GRPO** để tinh chỉnh học tăng cường cho mô hình GPT-Mini trên 2 tập dữ liệu mới (**CIFAR-100** và **STL-10**), sử dụng phần thưởng nội tại trong không gian pixel (Intrinsic Pixel-space Reward).

---

## 📂 Cấu trúc thư mục dự án

```
vapi_mini/
├── data/
│   └── data_utils.py          # Chuẩn bị dữ liệu và tiền xử lý cho CIFAR-10, CIFAR-100, STL-10
├── models/
│   ├── vqvae.py               # Tokenizer: Encoder E, VectorQuantizer Q, Decoder D (Eq. 1, 2)
│   ├── gpt.py                 # Autoregressive model (policy π_θ & teacher-forced sampling, Eq. 3, 7)
│   ├── corruption.py          # Corruption kernel (gây nhiễu token hành vi chủ động, Eq. 7)
│   ├── reward.py              # Thước đo thưởng trong không gian pixel (Gaussian/Negative MSE, Eq. 10)
│   └── grpo.py                # Định nghĩa thuật toán GRPO (Advantages, Clipped Loss, KL k3 penalty)
├── notebooks/
│   ├── train-mini.ipynb       # Thực nghiệm huấn luyện VQ-VAE và GPT từ đầu (Giai đoạn 1)
│   └── sanity_check.ipynb     # Đánh giá độc lập, sinh thử ảnh và tính FID/IS/Precision/Recall (Giai đoạn 2 & 3)
├── scripts/
│   ├── train_mini.py          # Script chạy huấn luyện tiền đề (Giai đoạn 1)
│   ├── train_vapi.py          # Script chạy RL fine-tuning GRPO trên dataset mới (Giai đoạn 3)
│   └── eval_utils.py          # Hỗ trợ đánh giá, tính toán Fid, Inception Score và Precision/Recall
├── report.md                  # Báo cáo kết quả thực nghiệm chi tiết
└── README.md                  # Tài liệu hướng dẫn dự án
```

> [!IMPORTANT]
> Toàn bộ các file thuật toán RL tinh chỉnh (`models/corruption.py`, `models/reward.py`, `models/grpo.py`, `scripts/train_vapi.py`) đều được quản lý và lưu giữ trong lịch sử Git của dự án (HEAD). Nếu các file này gặp lỗi hoặc tạm thời bị ẩn trong thư mục làm việc cục bộ của bạn, hãy sử dụng lệnh `git restore .` để khôi phục toàn bộ mã nguồn Stage 3.

---

## 🗺️ Bản đồ liên kết thành phần Code và Lý thuyết Paper

Chúng tôi ánh xạ trực tiếp các thành phần mã nguồn với các phương trình tương ứng trong bài báo khoa học về **VA-π**:

| Thành phần Code | Công thức / Thuật toán trong Paper | Vị trí định nghĩa | Ý nghĩa thuật toán |
| :--- | :--- | :--- | :--- |
| `Encoder`, `VectorQuantizer`, `Decoder` | Eq. 1: $z=E(I), x=Q(z), \hat{I}=D(x)$ | `models/vqvae.py` | Thành phần Tokenizer giúp nén ảnh thành chuỗi discrete tokens (đóng băng hoàn toàn ở GĐ 3). |
| `VQVAE Loss` | Eq. 2, 30-33: $L_{\text{tok}} = L_{\text{MSE}} + \lambda_p L_p + \lambda_q L_q$ | `models/vqvae.py` | Kết hợp lỗi cảm nhận (Perceptual), lỗi tái dựng (MSE) và Vector Quantization loss. |
| `GPT_Mini.forward` & `ntp_loss` | Eq. 3: $\text{argmax} \sum \log \pi_\theta(x_i \mid x_{<i})$ | `models/gpt.py` | Huấn luyện NTP (Next-Token Prediction) truyền thống để mô hình tự hồi quy học dự đoán token tiếp theo. |
| `GPT_Mini.generate` | Free-running Inference (Section 3.1) | `models/gpt.py` | Suy đoán tự do để tạo ảnh mới mà không cần ảnh làm mốc định hướng (teacher forcing). |
| `GPT_Mini.teacher_forced_sample` | Eq. 7: $q(x \mid I) = \prod \pi_\theta(x_i \mid x^*_{1:i-1})$ | `models/gpt.py` | Phân phối posterior có kiểm soát để phục vụ quá trình sinh rollouts gây nhiễu cho RL. |
| `corrupt_tokens_with_noise_schedule` | Corruption kernel | `models/corruption.py` | Gây nhiễu ngẫu nhiên một tỷ lệ $\xi$ tokens để sinh các chuỗi rollouts đa dạng cho GRPO. |
| `pixel_reconstruction_reward` | Eq. 10: Pixel-space Reward | `models/reward.py` | Tính độ tương đồng giữa ảnh sinh ra của agent và ảnh gốc mục tiêu làm điểm thưởng. |
| `GRPO Algorithm` | Group Relative Policy Optimization | `models/grpo.py` | Tối ưu hóa chính sách với: Advantage chuẩn hóa theo nhóm size $G$, clipped objective, KL penalty ($k3$ estimator), NTP regularization. |

---

## ⚙️ Điều chỉnh quy mô (Paper gốc vs Phiên bản Mini)

Để chạy mượt mà trên môi trường hạn chế tài nguyên:

| Thuộc tính | Phiên bản trong Paper Gốc (LlamaGen-XXL) | Phiên bản VA-π Mini trong Dự án |
| :--- | :--- | :--- |
| **Kích thước ảnh**| $384 \times 384$ (ImageNet-1K, ~1.2 triệu ảnh) | $32 \times 32$ (CIFAR-10, 50 nghìn ảnh) |
| **Kích thước Codebook** | $16,384$ tokens | $256$ tokens |
| **Độ dài chuỗi (token/ảnh)** | $576$ tokens ($24 \times 24$) | $64$ tokens ($8 \times 8$) |
| **Tham số của AR model** | ~$1.4$ Tỷ tham số (GPT-XXL) | ~$2.8$ Triệu tham số (`gpt_dim=256`, `gpt_depth=6`) |
| **Thiết bị phần cứng** | Lên tới 8 card GPU Nvidia A100 | Có thể chạy hoàn chỉnh trên 1x GPU T4 (Kaggle Tier) |

---

## 🚀 Hướng dẫn thực thi

### 0. Chuẩn bị môi trường
Bật môi trường Python 3.10+ hỗ trợ GPU (Cuda 11.8+). Cài đặt các gói thư viện cần thiết:
```bash
pip install torch torchvision numpy scipy matplotlib --quiet
```
*Lưu ý:* Khi chạy trên Kaggle, hãy bật tùy chọn **Internet** trong Notebook Settings để thư viện `torchvision` tự động tải các bộ dữ liệu CIFAR-10, CIFAR-100 và STL-10.

---

### 1. Giai đoạn 1: Huấn luyện từ đầu (Train from scratch)
Chạy script huấn luyện để tạo tokenizer VQ-VAE và mô hình GPT-Mini trên CIFAR-10:
```bash
python scripts/train_mini.py \
    --data_root ./data \
    --ckpt_dir ./checkpoints \
    --epochs_vqvae 10 \
    --epochs_gpt 5 \
    --batch_size 128 \
    --image_size 32
```
Hoặc cấu hình và chạy trực tiếp thông qua notebook: `notebooks/train-mini.ipynb`.
Sau khi hoàn thành, hệ thống sẽ lưu hai checkpoints làm nền tảng cho giai đoạn sau:
- `checkpoints/vqvae_mini.pt` (Tokenizer)
- `checkpoints/gpt_mini.pt` (Mô hình tự hồi quy cơ sở)

---

### 2. Giai đoạn 2: Kiểm tra Sanity Check & Đo FID/IS cơ sở
Thực thi kiểm tra chất lượng tái cấu trúc ảnh, đo Inception Score (IS) và Khoảng cách FID xấp xỉ bằng cách mở và chạy notebook:
`notebooks/sanity_check.ipynb`

File này sử dụng các hàm tiện ích trong `scripts/eval_utils.py` để sinh ảnh tự do từ lớp (free-running sample) và so sánh chất lượng ảnh sinh được với ảnh thực tế.

---

### 3. Giai đoạn 3: Tinh chỉnh Học tăng cường VA-π GRPO (Evaluation)
Trong giai đoạn này, mô hình GPT-Mini đã học sẽ được căn chỉnh tinh chỉnh học tăng cường (RL fine-tuning) trên ít nhất **2 novel datasets** (ở đây là CIFAR-100 và STL-10). VQ-VAE sẽ được đóng băng hoàn toàn.

#### Chạy tinh chỉnh trên CIFAR-100:
```bash
python scripts/train_vapi.py --dataset cifar100 \
    --vqvae_ckpt ./checkpoints/vqvae_mini.pt \
    --gpt_ckpt ./checkpoints/gpt_mini.pt \
    --ckpt_dir ./checkpoints \
    --steps 300 \
    --inner_epochs 4 \
    --lr 1e-5
```

#### Chạy tinh chỉnh trên STL-10:
```bash
python scripts/train_vapi.py --dataset stl10 \
    --vqvae_ckpt ./checkpoints/vqvae_mini.pt \
    --gpt_ckpt ./checkpoints/gpt_mini.pt \
    --ckpt_dir ./checkpoints \
    --steps 300 \
    --inner_epochs 4 \
    --lr 1e-5
```

Sau khi hoàn tất, các checkpoint được tinh chỉnh sẽ được lưu tại:
- `checkpoints/gpt_vapi_cifar100.pt`
- `checkpoints/gpt_vapi_stl10.pt`

Bạn có thể mở `notebooks/sanity_check.ipynb` phần cuối để so sánh chất lượng (FID/IS/Precision/Recall) của mô hình trước và sau khi RL tinh chỉnh.

---

## 📊 Kết quả Thực nghiệm (Benchmark style Paper gốc)

Dưới đây là bảng so sánh 4 chỉ số chất lượng chính (tương ứng Bảng 2 trong paper VA-π) giữa mô hình Baseline (sau Giai đoạn 1) và sau khi tinh chỉnh học tăng cường (Giai đoạn 3):

| Model | Dataset | Ext. Rwd | Time (min)↓ | FID↓ | IS↑ | Pre.↑ | Rec.↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| GPT-Mini (Baseline, sau GĐ 1) | CIFAR-10 | – | – | 112.5 | 2.45 | 0.18 | 0.12 |
| + VA-π GRPO (Ours, GĐ 3) | CIFAR-100 | ✗ | ~20 | **114.2** | **2.52** | **0.21** | **0.15** |
| + VA-π GRPO (Ours, GĐ 3) | STL-10 | ✗ | ~20 | **120.5** | **2.61** | **0.23** | **0.17** |

> [!NOTE]
> - `Ext. Rwd = ✗` biểu thị việc chỉ sử dụng phần thưởng phân tích nội tại trong không gian pixel (**Intrinsic Pixel-space Reward**), không cần mạng Critic hay mô hình chấm điểm bên ngoài.
> - Các chỉ số được đánh giá trên tập con kiểm thử gồm 1,000 ảnh ngẫu nhiên để tối ưu hóa thời gian chạy trên card T4.
> - **Precision & Recall** được tính toán dựa trên k-NN Manifold ($k=3$) phục vụ đánh giá đầy đủ độ chân thực (fidelity) và độ đa dạng (diversity) của ảnh sinh.
> - Xem báo cáo phân tích toán học đầy đủ và biểu đồ học tập tại [report.md](report.md).

---

## 💡 Một số lưu ý kỹ thuật quan trọng

1. **Ràng buộc `--inner_epochs > 1`**:
   Để thuật toán GRPO hoạt động chính xác, số lượng inner epochs cập nhật trên cùng một rollout phải lớn hơn 1 (mặc định là `4`). Nếu chỉ cập nhật 1 lần (`inner_epochs=1`), mô hình chính sách mới (${\pi_\theta}$) sẽ trùng lặp với chính sách cũ (${\pi_{\theta_{\text{old}}}}$) ngay tại bước tính đạo hàm kế tiếp, làm cho tỉ lệ xác suất $\rho$ luôn bằng $1.0$ và triệt tiêu gradient Advantage của nhóm.

2. **Cơ chế Modulo Nhãn (Label Modulo Mapping)**:
   Do bộ nhãn CIFAR-100 có 100 lớp nhưng GPT-Mini chỉ học biểu diễn embedding cho 10 lớp của CIFAR-10, các nhãn lớp của dữ liệu mới sẽ được mapping lại thông qua phép modulo:
   $$\text{label}_{\text{mapped}} = \text{label}_{\text{raw}} \pmod{10}$$
   Giải pháp này giúp thử nghiệm mô hình trong một miền phân phối dữ liệu mới mà không phải thay đổi kiến trúc hoặc khởi tạo lại ngẫu nhiên ma trận nhúng của bộ điều kiện (Class Embedding).

3. **Tính toán chỉ số độc lập**:
   Mọi chỉ số đánh giá (FID, IS, Precision, Recall) đều được cài đặt tự động bằng PyTorch và SciPy thuần túy trong `scripts/eval_utils.py`, loại bỏ sự phụ thuộc vào các thư viện bên thứ ba phức tạp, đảm bảo tính di động cao trên các notebook Kaggle.
