"""
Utility cho sanity-check Giai đoạn 2: kiểm tra checkpoint "Implement: Mini"
(VQVAE + GPT train from scratch trên CIFAR-10).

Gồm:
- load_checkpoints(): load lại VQVAE + GPT từ file .pt
- denormalize(): chuyển ảnh [-1,1] -> [0,1] để hiển thị/lưu PNG
- sample_and_decode(): sinh ảnh mới từ class label (free-running)
- compute_reconstruction_error(): MSE giữa ảnh gốc và ảnh reconstruct qua VQVAE
  (đo "sàn" chất lượng tokenizer — độc lập với GPT)
- compute_fid_is(): FID/IS xấp xỉ dùng InceptionV3 (torchvision), KHÔNG cần
  cài thêm package ngoài (pytorch-fid) để đơn giản hoá môi trường Kaggle.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from scipy import linalg

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vqvae import VQVAE
from models.gpt import GPT_Mini


def load_checkpoints(vqvae_path, gpt_path, device):
    """Load lại VQVAE + GPT đã train ở Giai đoạn 2, dựng đúng kiến trúc
    theo args đã lưu trong checkpoint (không cần nhớ lại thủ công)."""
    vq_ckpt = torch.load(vqvae_path, map_location=device, weights_only=False)
    vq_args = vq_ckpt["args"]
    vqvae = VQVAE(
        in_channels=3,
        hidden_dim=vq_args["hidden_dim_vq"],
        latent_dim=vq_args["latent_dim"],
        codebook_size=vq_args["codebook_size"],
        downsample_factor=vq_args["downsample_factor"],
    ).to(device)
    vqvae.load_state_dict(vq_ckpt["model_state_dict"])
    vqvae.eval()

    gpt_ckpt = torch.load(gpt_path, map_location=device, weights_only=False)
    gpt_args = gpt_ckpt["args"]
    gpt = GPT_Mini(
        vocab_size=gpt_args["codebook_size"],
        num_classes=gpt_ckpt["num_classes"],
        seq_len=gpt_ckpt["seq_len"],
        dim=gpt_args["gpt_dim"],
        depth=gpt_args["gpt_depth"],
        n_heads=gpt_args["gpt_heads"],
    ).to(device)
    gpt.load_state_dict(gpt_ckpt["model_state_dict"])
    gpt.eval()

    print(f"Đã load VQVAE: codebook={vq_args['codebook_size']}, "
          f"downsample={vq_args['downsample_factor']}")
    print(f"Đã load GPT: dim={gpt_args['gpt_dim']}, depth={gpt_args['gpt_depth']}, "
          f"seq_len={gpt_ckpt['seq_len']}, num_classes={gpt_ckpt['num_classes']}")
    return vqvae, gpt, gpt_ckpt["seq_len"]


def denormalize(images):
    """[-1, 1] -> [0, 1], clamp để tránh lỗi hiển thị do giá trị ngoài range."""
    return ((images + 1.0) / 2.0).clamp(0, 1)


@torch.no_grad()
def sample_and_decode(vqvae, gpt, class_ids, device, temperature=1.0, top_k=None):
    """Sinh ảnh mới hoàn toàn free-running: class_id -> token -> ảnh.
    Đây là cách kiểm tra TRUNG THỰC nhất chất lượng generative model,
    khác với reconstruction (vốn dùng token thật nên dễ hơn nhiều)."""
    class_ids = class_ids.to(device)
    gen_tokens = gpt.generate(class_ids, temperature=temperature, top_k=top_k)  # (B, N)
    B, N = gen_tokens.shape
    side = int(round(N ** 0.5))
    assert side * side == N, f"seq_len={N} không phải số chính phương, không reshape được lưới vuông"
    gen_tokens_grid = gen_tokens.reshape(B, side, side)
    images = vqvae.decode_from_indices(gen_tokens_grid)
    return denormalize(images)


@torch.no_grad()
def compute_reconstruction_error(vqvae, loader, device, max_batches=20):
    """MSE trung bình giữa ảnh gốc và ảnh reconstruct (encode rồi decode lại
    bằng đúng token đã encode, KHÔNG qua GPT). Đo riêng chất lượng tokenizer,
    tách biệt khỏi lỗi của GPT — hữu ích để debug xem lỗi nằm ở đâu."""
    vqvae.eval()
    total_mse, n = 0.0, 0
    for i, (images, _) in enumerate(loader):
        if i >= max_batches:
            break
        images = images.to(device)
        indices, _, _ = vqvae.encode(images)
        recon = vqvae.decode_from_indices(indices)
        total_mse += F.mse_loss(recon, images, reduction="sum").item()
        n += images.numel()
    return total_mse / n


# ----------------------------------------------------------------------------
# FID / IS xấp xỉ — dùng InceptionV3 có sẵn trong torchvision (không cần
# cài thêm package pytorch-fid để giữ môi trường Kaggle đơn giản).
# Lưu ý: đây là bản XẤP XỈ cho mục đích sanity-check nhanh trong Mini,
# KHÔNG phải FID chuẩn 50k-sample như paper gốc (cần ghi rõ trong báo cáo).
# ----------------------------------------------------------------------------
_inception_model = None


def _get_inception(device):
    global _inception_model
    if _inception_model is None:
        from torchvision.models import inception_v3
        model = inception_v3(weights="IMAGENET1K_V1", aux_logits=True)
        model.fc = torch.nn.Identity()  # lấy feature 2048-dim trước lớp phân loại
        model.eval().to(device)
        _inception_model = model
    return _inception_model


@torch.no_grad()
def get_inception_features(images, device, batch_size=64):
    """images: (N, 3, H, W) trong [0,1]. Trả về (N, 2048) feature."""
    model = _get_inception(device)
    # Inception cần input 299x299, chuẩn hoá theo ImageNet
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    feats = []
    for i in range(0, images.shape[0], batch_size):
        batch = images[i:i + batch_size].to(device)
        batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
        batch = (batch - mean) / std
        f = model(batch)
        feats.append(f.cpu())
    return torch.cat(feats, dim=0).numpy()


def compute_fid(real_features, fake_features):
    """FID chuẩn: khoảng cách Fréchet giữa 2 phân phối Gaussian khớp với feature thật/giả."""
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


@torch.no_grad()
def compute_inception_score(images, device, batch_size=64, splits=5):
    """IS xấp xỉ dùng logits phân loại 1000-class gốc của Inception
    (không phải feature 2048-dim, nên cần model riêng không bỏ .fc)."""
    from torchvision.models import inception_v3
    model = inception_v3(weights="IMAGENET1K_V1", aux_logits=True).eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    all_probs = []
    for i in range(0, images.shape[0], batch_size):
        batch = images[i:i + batch_size].to(device)
        batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
        batch = (batch - mean) / std
        logits = model(batch)
        probs = F.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)

    N = all_probs.shape[0]
    split_size = max(N // splits, 1)
    scores = []
    for k in range(splits):
        part = all_probs[k * split_size: (k + 1) * split_size]
        if len(part) == 0:
            continue
        py = part.mean(axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        kl = kl.sum(axis=1)
        scores.append(np.exp(kl.mean()))
    return float(np.mean(scores)), float(np.std(scores))
