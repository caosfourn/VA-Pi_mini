"""
Reward function cho VA-π (Eq. 10 trong paper).

Sau khi sample token mới x_sample (từ teacher-forced logits trên noisy
context), token này được decode thành ảnh I_hat = D(Q_lookup(x_sample)).
Reward được định nghĩa dựa trên độ giống nhau pixel-space giữa I_hat và
ảnh tham chiếu thật I (KHÔNG cần reward model bên ngoài — đây là điểm
mạnh "intrinsic reward" mà paper nhấn mạnh trong abstract).

Paper dùng công thức dạng:
    r = exp(-||I_hat - I||^2 / tau)     (similarity dạng Gaussian kernel)
hoặc đơn giản hơn là âm của MSE/LPIPS. Bản Mini này cung cấp cả 2 lựa
chọn (negative MSE và Gaussian-kernel reward) để linh hoạt, mặc định
dùng Gaussian-kernel vì cho reward nằm trong [0,1] — dễ chuẩn hoá hơn
khi đưa vào GRPO advantage.
"""

import torch
import torch.nn.functional as F


def pixel_reconstruction_reward(generated_images: torch.Tensor,
                                 reference_images: torch.Tensor,
                                 mode: str = "gaussian",
                                 tau: float = 0.5) -> torch.Tensor:
    """
    Tính reward pixel-space giữa ảnh decode từ token sample (generated_images)
    và ảnh thật tham chiếu (reference_images). Cả hai cùng range [-1, 1].

    generated_images, reference_images: (B, 3, H, W)
    mode:
        "gaussian"  -> r = exp(-MSE / tau)   (Eq. 10 dạng kernel, r in (0,1])
        "neg_mse"   -> r = -MSE              (đơn giản, không chặn range)
    tau: hệ số temperature cho Gaussian kernel — tau nhỏ làm reward nhạy
         hơn với sai khác nhỏ (phân biệt rõ ràng hơn giữa good/bad sample).

    Trả về: (B,) reward cho mỗi sample trong batch.
    """
    B = generated_images.shape[0]
    mse_per_sample = F.mse_loss(
        generated_images, reference_images, reduction="none"
    ).reshape(B, -1).mean(dim=1)  # (B,)

    if mode == "gaussian":
        reward = torch.exp(-mse_per_sample / tau)
    elif mode == "neg_mse":
        reward = -mse_per_sample
    else:
        raise ValueError(f"Unknown reward mode: {mode}")

    return reward


def perceptual_reward(perceptual_net, generated_images, reference_images,
                       mode: str = "gaussian", tau: float = 1.0):
    """
    Biến thể reward dùng feature-space distance (qua SimplePerceptualNet đã
    có trong vqvae.py) thay vì pixel MSE thuần — bắt được sự khác biệt về
    cấu trúc/texture tốt hơn MSE đơn thuần. Tùy chọn, không bắt buộc dùng.
    """
    B = generated_images.shape[0]
    f_gen = perceptual_net(generated_images)
    f_ref = perceptual_net(reference_images)
    dist_per_sample = F.mse_loss(f_gen, f_ref, reduction="none").reshape(B, -1).mean(dim=1)

    if mode == "gaussian":
        reward = torch.exp(-dist_per_sample / tau)
    elif mode == "neg_mse":
        reward = -dist_per_sample
    else:
        raise ValueError(f"Unknown reward mode: {mode}")
    return reward
