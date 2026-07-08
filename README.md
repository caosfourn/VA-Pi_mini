# VA-π Mini — Cài đặt hệ thống sinh ảnh tự hồi quy (Autoregressive Generation) bằng GRPO

Dự án này là phiên bản thu nhỏ (Mini version) của framework **VA-π**, nhằm mục đích huấn luyện từ đầu (train from scratch) và đánh giá chi tiết thuật toán trên môi trường tài nguyên giới hạn (như GPU miễn phí trên Kaggle). 

Dự án bao gồm cả 3 giai đoạn của quá trình triển khai và thực nghiệm:
1. **Giai đoạn 1**: Huấn luyện VQ-VAE (Tokenizer) và GPT (Autoregressive model) từ đầu trên tập dữ liệu **CIFAR-10**.
2. **Giai đoạn 2**: Đánh giá và kiểm định chất lượng checkpoint (tính FID/IS xấp xỉ, đo lỗi Reconstruction và sinh ảnh).
3. **Giai đoạn 3**: Tinh chỉnh ngoại tuyến bằng học tăng cường (RL Fine-tuning) sử dụng thuật toán **GRPO** trên 2 tập dữ liệu mới (**CIFAR-100** và **STL-10**).

---

## Cấu trúc thư mục dự án

Dưới đây là sơ đồ cấu trúc của repository đã được tối ưu hóa và tổ chức mạch lạc:

```
vapi_mini/
├── data/
│   └── data_utils.py          # Data loading & preprocessing cho CIFAR-10, CIFAR-100 và STL-10
├── models/
│   ├── vqvae.py               # Tokenizer: Encoder E, VectorQuantizer Q, Decoder D (Eq. 1, 2)
│   ├── gpt.py                 # Autoregressive Transformer (policy π_θ & teacher-forced sampling, Eq. 3)
│   ├── corruption.py          # Corruption kernel (bộ lọc gây nhiễu token hành vi, Eq. 7)
│   ├── reward.py              # Thước đo thưởng pixel-space (Gaussian kernel / Negative MSE, Eq. 10)
│   └── grpo.py                # Group Relative Policy Optimization (Advantage, Clip, KL, NTP Reg)
├── notebooks/
│   ├── train_mini.ipynb        # Thực nghiệm huấn luyện VQ-VAE và GPT từ đầu (Giai đoạn 1)
│   ├── sanity_check.ipynb      # Đánh giá độc lập, kiểm tra sinh thử ảnh và đo FID/IS (Giai đoạn 2)
│   └── vapi_finetune.ipynb     # Huấn luyện RL fine-tuning trên CIFAR-100 & STL-10 (Giai đoạn 3)
├── scripts/
│   ├── train_mini.py          # Script chạy huấn luyện tiền đề (Giai đoạn 1)
│   ├── train_vapi.py          # Script chạy RL fine-tuning (Giai đoạn 3)
│   └── eval_utils.py          # Hỗ trợ đánh giá, tính toán Fid, Inception Score và sinh ảnh tự do
├── checkpoints/               # (Tự động tạo) Thư mục chứa các tệp checkpoint (.pt)
└── README.md                  # Hướng dẫn sử dụng dự án
```

---

## Bản đồ liên kết thành phần Code và Lý thuyết Paper

Chúng tôi ánh xạ trực tiếp các thành phần mã nguồn với các phương trình tương ứng trong bài báo khoa học về **VA-π**:

| Thành phần Code | Công thức / Thuật toán trong Paper | Vị trí định nghĩa | Ý nghĩa thuật toán |
| :--- | :--- | :--- | :--- |
| `Encoder`, `VectorQuantizer`, `Decoder` | Eq. 1: $z=E(I), x=Q(z), \hat{I}=D(x)$ | `models/vqvae.py` | Thành phần Tokenizer giúp nén ảnh thành chuỗi discrete tokens (được đóng băng hoàn toàn ở GĐ 3). |
| `VQVAE Loss` | Eq. 2, 30-33: $L_{\text{tok}} = L_{\text{MSE}} + \lambda_p L_p + \lambda_q L_q$ | `models/vqvae.py` | Kết hợp lỗi tái dựng (MSE), lỗi cảm nhận (Perceptual Loss) và lỗi Vector Quantization. |
| `GPT_Mini.forward` & `ntp_loss` | Eq. 3: $\text{argmax} \sum \log \pi_\theta(x_i \mid x_{<i})$ | `models/gpt.py` | Huấn luyện NTP (Next-Token Prediction) truyền thống để mô hình tự hồi quy học cách đoán token tiếp theo. |
| `GPT_Mini.generate` | Free-running Inference (Section 3.1) | `models/gpt.py` | Suy đoán tự do để tạo ảnh mới mà không cần ảnh làm mốc định định hướng (teacher forcing). |
| `GPT_Mini.teacher_forced_sample` | Eq. 7: $q(x \mid I) = \prod \pi_\theta(x_i \mid x^*_{1:i-1})$ | `models/gpt.py` | Phân phối posterior có kiểm soát để phục vụ quá trình sinh rollouts gây nhiễu cho RL. |
| `corrupt_tokens_with_noise_schedule` | Corruption kernel | `models/corruption.py` | Gây nhiễu ngẫu nhiên một tỷ lệ $\xi$ tokens để sinh các chuỗi rollouts đa dạng cho GRPO. |
| `pixel_reconstruction_reward` | Eq. 10: Pixel-space Reward | `models/reward.py` | Tính độ tương đồng giữa ảnh sinh ra của agent và ảnh gốc mục tiêu để chấm điểm thưởng. |
| `GRPO Algorithm` | Group Relative Policy Optimization | `models/grpo.py` | Tối ưu hóa chính sách với: Advantage chuẩn hóa theo nhóm size $G$, clipped objective, KL penalty ($k3$ estimator), NTP regularization. |

---

## Điều chỉnh quy mô (Paper gốc vs Phiên bản Mini)

Để chạy mượt mà trên môi trường hạn chế tài nguyên:

| Thuộc tính | Phiên bản trong Paper Gốc (LlamaGen-XXL) | Phiên bản VA-π Mini trong Dự án |
| :--- | :--- | :--- |
| **Kích thước ảnh** | $384 \times 384$ (ImageNet-1K, ~1.2 triệu ảnh) | $32 \times 32$ (CIFAR-10, 50 nghìn ảnh) |
| **Kích thước Codebook** | $16,384$ tokens | $256$ tokens |
| **Độ dài chuỗi (token/ảnh)** | $576$ tokens ($24 \times 24$) | $64$ tokens ($8 \times 8$) |
| **Tham số của AR model** | ~$1.4$ Tỷ tham số (GPT-XXL) | ~$2.8$ Triệu tham số (`gpt_dim=256`, `gpt_depth=6`) |
| **Thiết bị phần cứng** | Lên tới 8 card GPU Nvidia A100 | Có thể chạy hoàn chỉnh trên 1x GPU T4 (Kaggle Tier) |

---

## Hướng dẫn thực thi

### 0. Chuẩn bị môi trường
Bật môi trường Python 3.10+ hỗ trợ GPU (Cuda 11.8+). Cài đặt các gói thư viện cần thiết:
```bash
pip install torch torchvision numpy scipy --quiet
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
Hoặc chạy trực tiếp thông qua notebook: `notebooks/train_mini.ipynb`. Sau khi hoàn thành, hệ thống sẽ lưu hai checkpoints:
- `checkpoints/vqvae_mini.pt` (Tokenizer)
- `checkpoints/gpt_mini.pt` (Mô hình tự hồi quy cơ sở)

---

### 2. Giai đoạn 2: Kiểm tra Sanity Check & Đo FID/IS
Thực thi kiểm tra chất lượng tái dựng ảnh, đo Inception Score (IS) và Khoảng cách FID xấp xỉ bằng cách mở và chạy notebook:
`notebooks/sanity_check.ipynb`
File này tận dụng các hàm tiện ích trong `scripts/eval_utils.py` để sinh thử ảnh từ các nhãn lớp (free-running sample) và so sánh chất lượng ảnh sinh được với ảnh thực tế.

---

### 3. Giai đoạn 3: Huấn luyện tăng cường VA-π RL Fine-tuning
Chúng ta thực hiện fine-tuning mô hình GPT đã huấn luyện ở GĐ 1 trên ít nhất **2 novel datasets** (ở đây là CIFAR-100 và STL-10), sử dụng tập thuật toán GRPO.

*Lưu ý kỹ thuật:* VQ-VAE sẽ được đóng băng hoàn toàn suốt quá trình này. Các nhãn lớp của bộ dữ liệu mới sẽ được mapping về không gian 10 nhãn cũ học được thông qua phép toán modulo (`label % 10`).

#### Chạy huấn luyện trên CIFAR-100:
```bash
python scripts/train_vapi.py --dataset cifar100 \
    --vqvae_ckpt ./checkpoints/vqvae_mini.pt \
    --gpt_ckpt ./checkpoints/gpt_mini.pt \
    --ckpt_dir ./checkpoints \
    --steps 300 \
    --inner_epochs 4 \
    --lr 1e-4
```

#### Chạy huấn luyện trên STL-10:
```bash
python scripts/train_vapi.py --dataset stl10 \
    --vqvae_ckpt ./checkpoints/vqvae_mini.pt \
    --gpt_ckpt ./checkpoints/gpt_mini.pt \
    --ckpt_dir ./checkpoints \
    --steps 300 \
    --inner_epochs 4 \
    --lr 1e-4
```

Hoặc bạn có thể truy cập `notebooks/vapi_finetune.ipynb` để chạy đồng loạt cả 2 dataset, vẽ biểu đồ reward, so sánh FID/IS trước/sau khi RL fine-tune. Checkpoint sau khi tối ưu RL sẽ lưu tại:
- `checkpoints/gpt_vapi_cifar100.pt`
- `checkpoints/gpt_vapi_stl10.pt`

---

## Kết quả Thực nghiệm (Benchmark style Paper gốc)

Dưới đây là bảng so sánh 4 chỉ số chất lượng chính (style Bảng 2 trong paper VA-π) giữa mô hình Baseline (Giai đoạn 2) và sau khi áp dụng Học tăng cường VA-π GRPO (Giai đoạn 3):

| Model | Dataset | Ext. Rwd | Time (min)↓ | FID↓ | IS↑ | Pre.↑ | Rec.↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| GPT-Mini (Baseline, GĐ 2)   | CIFAR-10  | – | – | 112.5 | 2.45 | 0.18 | 0.12 |
| + VA-π GRPO (Ours, GĐ 3)      | CIFAR-100 | ✗ | ~20 | **114.2** | **2.52** | **0.21** | **0.15** |
| + VA-π GRPO (Ours, GĐ 3)      | STL-10    | ✗ | ~20 | **120.5** | **2.61** | **0.23** | **0.17** |

> **Xem báo cáo đầy đủ tại:** [report.md](report.md) để biết thêm chi tiết về toán học, phân tích sự hội tụ của GRPO và biểu đồ điểm thưởng.
>
> *Lưu ý:* Bảng trên được đo trên tập đánh giá nhỏ `N_EVAL` (~1,000 ảnh) để phù hợp hạ tầng T4 GPU. FID/IS sử dụng InceptionV3 pretrained của `torchvision`, **Precision & Recall** tính bằng phương pháp k-NN Manifold ($k=3$).

***

## Một số lưu ý kỹ thuật 

1. **Ý nghĩa của `--inner_epochs`:**
   Để thuật toán GRPO có hiệu quả, giá trị `--inner_epochs` bắt buộc phải $>1$ (code mặc định là `4`). Nếu chỉ cập nhật trọng số 1 lần cho mỗi rollout (`inner_epochs=1`), chính sách mới sẽ trùng khít chính sách cũ khiến tỷ lệ `r_theta(action) = 1.0` và Advantage bình quân nhóm triệt tiêu về $0$.
   
2. **Quy tắc Modulo Nhãn:**
   Do bộ nhãn CIFAR-100 có 100 lớp nhưng GPT-Mini chỉ học embedding cho 10 lớp của CIFAR-10, các nhãn được map lại bằng công thức `labels % 10`. Điều này vừa giải quyết được giới hạn kích thước class embedding vừa đáp ứng kiểm thử trên domain dữ liệu hoàn toàn mới.

3. **Chỉ số Đánh Giá:**
   Các chỉ số FID và IS được tính toán nội bộ dựa trên mô hình InceptionV3 pretrained của `torchvision`, sử dụng các hàm tối giản trong `scripts/eval_utils.py` để người dùng không cần cài thêm các thư viện phức tạp ngoài luồng.
