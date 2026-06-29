"""
VQ-VAE Tokenizer cho VA-π (Mini implementation).

Tương ứng với Sec 3.1 của paper:
    z = E(I),  x = Q(z),  I_hat = D(x)                       (Eq. 1)

Tokenizer loss (Eq. 2, Eq. 30-33):
    L_tok = L_MSE + lambda_p * L_p + lambda_q * L_q

Đây là bản THU NHỎ của VQGAN dùng trong LlamaGen gốc — phù hợp để
train from scratch trên ảnh nhỏ (32x32 hoặc 64x64) trong thời gian
giới hạn của Kaggle, nhưng giữ đúng 3 thành phần: Encoder E, Quantizer Q
(codebook), Decoder D, và đúng 3 loss thành phần (MSE + Perceptual + VQ).

Vai trò trong toàn bộ pipeline VA-π:
- Giai đoạn "Implement: Mini": tokenizer này được TRAIN TRƯỚC (hoặc cùng) với
  AR model trên dataset gốc.
- Giai đoạn "Evaluation" (VA-π RL fine-tuning): tokenizer này bị ĐÓNG BĂNG
  (frozen) — paper nói rõ "we only update the AR generator πθ while keeping
  the tokenizer ϕ, ψ frozen" (Sec 4.1). Encoder/Quantizer dùng để tạo
  ground-truth tokens x*, Decoder dùng để tính reward pixel-space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Residual block dùng trong Encoder/Decoder
# ----------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


# ----------------------------------------------------------------------------
# Encoder E: I (3xHxW) -> z (C x N), N = (H/downsample) * (W/downsample)
# ----------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=128, latent_dim=64,
                 downsample_factor=4, num_res_blocks=2):
        super().__init__()
        assert downsample_factor in (2, 4, 8), "downsample_factor phải là 2, 4 hoặc 8"
        num_down = {2: 1, 4: 2, 8: 3}[downsample_factor]

        layers = [nn.Conv2d(in_channels, hidden_dim, 3, padding=1)]
        ch = hidden_dim
        for _ in range(num_down):
            layers += [nn.Conv2d(ch, ch * 2, 4, stride=2, padding=1), nn.SiLU()]
            ch *= 2
            for _ in range(num_res_blocks):
                layers.append(ResBlock(ch))
        layers += [
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, latent_dim, 1),  # chiếu về latent_dim (= C trong paper)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # (B, C, H', W')


# ----------------------------------------------------------------------------
# Decoder D: x (token indices) -> I_hat (3xHxW)
# ----------------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, out_channels=3, hidden_dim=128, latent_dim=64,
                 downsample_factor=4, num_res_blocks=2):
        super().__init__()
        num_up = {2: 1, 4: 2, 8: 3}[downsample_factor]
        ch = hidden_dim * (2 ** num_up)

        layers = [nn.Conv2d(latent_dim, ch, 1), nn.SiLU()]
        for _ in range(num_up):
            for _ in range(num_res_blocks):
                layers.append(ResBlock(ch))
            layers += [
                nn.ConvTranspose2d(ch, ch // 2, 4, stride=2, padding=1),
                nn.SiLU(),
            ]
            ch //= 2
        layers += [
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
            nn.Tanh(),  # output trong [-1, 1]
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------------
# Vector Quantizer Q: ánh xạ z liên tục -> token rời rạc qua codebook
# Tương ứng Eq. 33: L_q = ||sg[z]-e||^2 + beta*||z-sg[e]||^2
# ----------------------------------------------------------------------------
class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int = 512, latent_dim: int = 64, beta: float = 0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.beta = beta
        self.codebook = nn.Embedding(codebook_size, latent_dim)
        # init giống VQGAN gốc: uniform nhỏ
        self.codebook.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z):
        """
        z: (B, C, H, W) -> trả về (z_q, indices, vq_loss)
        z_q đã áp Straight-Through Estimator để gradient chảy qua Encoder.
        """
        B, C, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)

        # khoảng cách L2 tới từng codebook vector
        d = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1)
        )
        indices = torch.argmin(d, dim=1)  # (B*H*W,)
        z_q_flat = self.codebook(indices)  # (B*H*W, C)

        # Eq. 33: commitment + codebook loss
        codebook_loss = F.mse_loss(z_q_flat, z_flat.detach())
        commitment_loss = F.mse_loss(z_q_flat.detach(), z_flat)
        vq_loss = codebook_loss + self.beta * commitment_loss

        # Straight-Through Estimator: forward dùng z_q, backward dùng gradient của z
        z_q_flat = z_flat + (z_q_flat - z_flat).detach()

        z_q = z_q_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        indices = indices.reshape(B, H, W)
        return z_q, indices, vq_loss

    def lookup(self, indices):
        """indices: (B, H, W) hoặc (B, N) long tensor -> z_q tương ứng (dùng khi decode từ token đã sample)."""
        z_q = self.codebook(indices)  # (..., C)
        if indices.dim() == 3:
            z_q = z_q.permute(0, 3, 1, 2)  # (B, C, H, W)
        return z_q


# ----------------------------------------------------------------------------
# Perceptual loss L_p (Eq. 32) — dùng vài lớp đầu của một CNN nhỏ thay cho
# VGG đầy đủ (VGG pretrained nặng, không cần thiết cho ảnh nhỏ 32x32/64x64).
# Để giữ tinh thần "có loss perceptual" mà vẫn nhẹ cho Kaggle.
# ----------------------------------------------------------------------------
class SimplePerceptualNet(nn.Module):
    """CNN nhỏ, KHÔNG cần pretrained, dùng làm feature extractor cho L_p.
    Đây là lựa chọn rút gọn hợp lý cho bản Mini; nếu muốn trung thành 100%
    với paper có thể thay bằng `torchvision.models.vgg16(pretrained=True)`."""

    def __init__(self, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        for p in self.parameters():
            p.requires_grad_(False)  # cố định, không train cùng VQVAE

    def forward(self, x):
        return self.features(x)


def perceptual_loss(perceptual_net, img1, img2):
    """L_p (Eq. 32): tổng MSE trên feature map của 1 mạng phụ trợ."""
    f1 = perceptual_net(img1)
    f2 = perceptual_net(img2)
    return F.mse_loss(f1, f2)


# ----------------------------------------------------------------------------
# VQVAE đầy đủ: kết hợp Encoder + Quantizer + Decoder + loss tổng (Eq. 2)
# ----------------------------------------------------------------------------
class VQVAE(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=128, latent_dim=64,
                 codebook_size=512, downsample_factor=4, beta=0.25,
                 lambda_p=1.0):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_dim, latent_dim, downsample_factor)
        self.quantizer = VectorQuantizer(codebook_size, latent_dim, beta)
        self.decoder = Decoder(in_channels, hidden_dim, latent_dim, downsample_factor)
        self.perceptual_net = SimplePerceptualNet(in_channels)
        self.lambda_p = lambda_p
        self.downsample_factor = downsample_factor
        self.codebook_size = codebook_size

    def encode(self, images):
        """I -> (indices, z_q). Dùng để lấy ground-truth tokens x* = Q(E(I))."""
        z = self.encoder(images)
        z_q, indices, vq_loss = self.quantizer(z)
        return indices, z_q, vq_loss

    def decode_from_indices(self, indices):
        """token indices -> ảnh decode I_hat. Dùng cho reward VA-π (Eq. 10)."""
        z_q = self.quantizer.lookup(indices)
        return self.decoder(z_q)

    def forward(self, images):
        """Forward đầy đủ cho training tokenizer: trả ảnh reconstruct + các loss."""
        indices, z_q, vq_loss = self.encode(images)
        recon = self.decoder(z_q)

        l_mse = F.mse_loss(recon, images)
        l_p = perceptual_loss(self.perceptual_net, recon, images)

        # Eq. 2: L_tok = L_MSE + lambda_p * L_p + lambda_q * L_q
        # (vq_loss đã gộp commitment*beta nên đóng vai trò lambda_q*L_q ở đây)
        total_loss = l_mse + self.lambda_p * l_p + vq_loss

        loss_dict = {
            "loss": total_loss,
            "l_mse": l_mse.detach(),
            "l_p": l_p.detach(),
            "l_q": vq_loss.detach(),
        }
        return recon, indices, loss_dict

    @property
    def token_grid_size(self):
        """Số token theo 1 chiều (N = grid*grid)."""
        return None  # set tùy theo image_size / downsample_factor khi khởi tạo dataset
