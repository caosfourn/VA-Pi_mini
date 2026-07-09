"""
Utility cho sanity-check Giai đoạn 2: kiểm tra checkpoint "Implement: Mini"
(VQVAE + GPT train from scratch trên CIFAR-10).

Gồm:
- load_checkpoints()          : load lại VQVAE + GPT từ file .pt
- denormalize()               : chuyển ảnh [-1,1] -> [0,1] để hiển thị/lưu PNG
- sample_and_decode()         : sinh ảnh mới từ class label (free-running)
- compute_reconstruction_error(): MSE giữa ảnh gốc và ảnh reconstruct qua VQVAE
                                  (đo "sàn" chất lượng tokenizer — độc lập với GPT)
- compute_fid()               : FID xấp xỉ dùng InceptionV3 (torchvision)
- compute_inception_score()   : IS xấp xỉ dùng InceptionV3 (torchvision)
- compute_precision_recall()  : Precision & Recall theo định nghĩa k-NN manifold
                                  (Kynkäänniemi et al. 2019), đúng với paper VA-π gốc.
- compute_all_metrics()       : Hàm tổng hợp — trả về FID, IS, Pre., Rec. một lần,
                                  dùng để điền bảng benchmark style paper.

Lưu ý: Tất cả metric đều là XẤP XỈ trên tập nhỏ CIFAR,
KHÔNG phải chuẩn 50k-sample như paper gốc (cần ghi rõ trong báo cáo).
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
        out = model(batch)
        f = out.logits if hasattr(out, 'logits') else out[0]  # InceptionV3 aux_logits returns tuple
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
        out = model(batch)
        logits = out.logits if hasattr(out, 'logits') else out[0]  # InceptionV3 aux_logits returns tuple
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


# ----------------------------------------------------------------------------
# Precision & Recall — k-NN Manifold (Kynkäänniemi et al. 2019)
# Định nghĩa: ảnh sinh ra nằm trong "manifold" thật (Precision đo fidelity),
# và ảnh thật nằm trong "manifold" sinh ra (Recall đo diversity).
# Đây là 2 chỉ số còn thiếu để hoàn thiện bảng benchmark như Bảng 2 trong paper.
# ----------------------------------------------------------------------------

def _compute_knn_radii(features: np.ndarray, k: int) -> np.ndarray:
    """Với mỗi điểm trong features, tính khoảng cách tới hàng xóm thứ k.
    Dùng thuật toán brute-force O(N^2) — phù hợp cho N <= 5000 như CIFAR mini."""
    # Tính ma trận khoảng cách bình phương bằng dot-product trick để tận dụng numpy
    sq = (features ** 2).sum(axis=1, keepdims=True)          # (N, 1)
    D2 = sq + sq.T - 2 * (features @ features.T)             # (N, N)
    D2 = np.maximum(D2, 0.0)                                  # clip lỗi số học nhỏ
    # Với mỗi hàng, lấy phần tử thứ k+1 (index k sau khi sort) — bỏ qua chính nó
    radii_sq = np.partition(D2, k + 1, axis=1)[:, k]         # (N,)  — k-th neighbor
    return np.sqrt(np.maximum(radii_sq, 0.0))                 # (N,)  — khoảng cách


def compute_precision_recall(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    k: int = 3,
) -> tuple[float, float]:
    """Tính Precision và Recall theo manifold k-NN.

    Precision = tỷ lệ ảnh giả nằm trong Manifold thật  (đo fidelity / chất lượng)
    Recall    = tỷ lệ ảnh thật nằm trong Manifold giả  (đo diversity / độ phủ)

    Args:
        real_features : (N_r, D) — Inception features của ảnh thật.
        fake_features : (N_f, D) — Inception features của ảnh sinh.
        k             : số hàng xóm cho manifold (mặc định k=3, theo paper gốc).

    Returns:
        (precision, recall) — hai float trong [0, 1].
    """
    # Tính bán kính manifold cho từng tập
    real_radii = _compute_knn_radii(real_features, k)  # (N_r,)
    fake_radii = _compute_knn_radii(fake_features, k)  # (N_f,)

    # Precision: với mỗi điểm giả, kiểm tra có hàng xóm thật nào
    # trong bán kính real_radii của hàng xóm thật đó không
    # <=> khoảng cách(fake_i, real_j) <= real_radii[j]  với ít nhất 1 j
    D_fr = np.sqrt(np.maximum(
        (fake_features ** 2).sum(1, keepdims=True)
        + (real_features ** 2).sum(1)
        - 2 * (fake_features @ real_features.T),
        0.0,
    ))  # (N_f, N_r)
    in_real_manifold = (D_fr <= real_radii[np.newaxis, :]).any(axis=1)  # (N_f,)
    precision = float(in_real_manifold.mean())

    # Recall: với mỗi điểm thật, kiểm tra có hàng xóm giả nào
    # trong bán kính fake_radii của hàng xóm giả đó không
    # <=> khoảng cách(real_i, fake_j) <= fake_radii[j]  với ít nhất 1 j
    D_rf = D_fr.T  # (N_r, N_f)  — transpose để tái dùng, không tính lại
    in_fake_manifold = (D_rf <= fake_radii[np.newaxis, :]).any(axis=1)  # (N_r,)
    recall = float(in_fake_manifold.mean())

    return precision, recall


# ----------------------------------------------------------------------------
# Hàm tổng hợp — trả về đầy đủ bảng benchmark như trong paper VA-π (Bảng 2)
# ----------------------------------------------------------------------------

@torch.no_grad()
def compute_all_metrics(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device,
    inception_batch_size: int = 64,
    is_splits: int = 5,
    knn_k: int = 3,
) -> dict:
    """Tính đầy đủ FID, IS (mean ± std), Precision, Recall trong một lần gọi.

    Args:
        real_images          : (N, 3, H, W) tensor ảnh thật trong [0, 1].
        fake_images          : (M, 3, H, W) tensor ảnh sinh trong [0, 1].
        device               : torch device.
        inception_batch_size : batch size khi chạy InceptionV3 (giảm nếu OOM).
        is_splits            : số splits khi tính IS.
        knn_k                : k cho manifold Precision/Recall.

    Returns:
        dict với các key: 'fid', 'is_mean', 'is_std', 'precision', 'recall'.

    Ví dụ sử dụng:
        metrics = compute_all_metrics(real_imgs, fake_imgs, device)
        print(f"FID={metrics['fid']:.2f}  IS={metrics['is_mean']:.2f}±{metrics['is_std']:.2f}"
              f"  Pre={metrics['precision']:.3f}  Rec={metrics['recall']:.3f}")
    """
    print("[compute_all_metrics] Trích xuất Inception features cho ảnh thật...")
    real_feats = get_inception_features(real_images, device, batch_size=inception_batch_size)

    print("[compute_all_metrics] Trích xuất Inception features cho ảnh sinh...")
    fake_feats = get_inception_features(fake_images, device, batch_size=inception_batch_size)

    print("[compute_all_metrics] Tính FID...")
    fid = compute_fid(real_feats, fake_feats)

    print("[compute_all_metrics] Tính Inception Score (IS)...")
    is_mean, is_std = compute_inception_score(
        fake_images, device,
        batch_size=inception_batch_size,
        splits=is_splits,
    )

    print(f"[compute_all_metrics] Tính Precision & Recall (k={knn_k})...")
    precision, recall = compute_precision_recall(real_feats, fake_feats, k=knn_k)

    results = {
        "fid"       : round(fid, 4),
        "is_mean"   : round(is_mean, 4),
        "is_std"    : round(is_std, 4),
        "precision" : round(precision, 4),
        "recall"    : round(recall, 4),
    }

    # In bảng tóm tắt ngay trong console để tiện đọc khi chạy notebook
    print("\n" + "=" * 52)
    print(f"{'Metric':<20} {'Value':>10}")
    print("-" * 52)
    print(f"{'FID ↓':<20} {results['fid']:>10.4f}")
    print(f"{'IS ↑ (mean)':<20} {results['is_mean']:>10.4f}")
    print(f"{'IS ↑ (std)':<20} {results['is_std']:>10.4f}")
    print(f"{'Precision ↑':<20} {results['precision']:>10.4f}")
    print(f"{'Recall ↑':<20} {results['recall']:>10.4f}")
    print("=" * 52)

    return results
