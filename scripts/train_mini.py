"""
Script train "Implement: Mini" cho VA-π.

Đáp ứng yêu cầu rubric:
    "Mini: Training the model from scratch on tested datasets for at least 1 epoch"

Pipeline gồm 2 bước, train from scratch hoàn toàn (không dùng checkpoint pretrained):
    Bước 1: Train VQVAE tokenizer (Encoder/Quantizer/Decoder) trên CIFAR-10
            -> tối ưu Eq. 2: L_tok = L_MSE + lambda_p*L_p + lambda_q*L_q
    Bước 2: Dùng VQVAE đã train (đóng băng) để encode toàn bộ ảnh CIFAR-10
            thành token, sau đó train GPT_Mini (AR transformer) bằng
            next-token-prediction loss (Eq. 3) ít nhất 1 epoch.

Checkpoint của cả 2 bước được lưu lại — đây chính là "model đã Implement: Mini"
sẽ được dùng làm điểm khởi đầu cho giai đoạn Evaluation (VA-π RL fine-tuning
trên >=2 dataset mới, xem evaluate_vapi.py).

Cách chạy (trên Kaggle, có GPU):
    python train_mini.py --epochs_vqvae 10 --epochs_gpt 5 --image_size 32

Lưu ý quy mô (đã thu nhỏ có chủ đích so với paper gốc để chạy được trên
1-2 GPU Kaggle trong vài giờ):
    - Ảnh 32x32 (CIFAR-10 gốc), downsample x4 -> 8x8 = 64 token/ảnh
    - Codebook 256 (so với 16384 của LlamaGen gốc)
    - GPT dim=256, depth=6 (so với GPT-XXL ~1.4B params của LlamaGen)
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vqvae import VQVAE
from models.gpt import GPT_Mini, ntp_loss
from data.data_utils import get_cifar10_loaders


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--train_subset", type=int, default=None,
                    help="Giới hạn số ảnh train (debug nhanh). None = dùng full CIFAR-10 (50k ảnh).")

    # VQVAE
    p.add_argument("--codebook_size", type=int, default=256)
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--hidden_dim_vq", type=int, default=64)
    p.add_argument("--downsample_factor", type=int, default=4)
    p.add_argument("--epochs_vqvae", type=int, default=10)
    p.add_argument("--lr_vqvae", type=float, default=2e-4)

    # GPT
    p.add_argument("--gpt_dim", type=int, default=256)
    p.add_argument("--gpt_depth", type=int, default=6)
    p.add_argument("--gpt_heads", type=int, default=8)
    p.add_argument("--epochs_gpt", type=int, default=5)
    p.add_argument("--lr_gpt", type=float, default=3e-4)

    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def train_vqvae(args, train_loader, device):
    print("\n" + "=" * 60)
    print("BƯỚC 1/2: Train VQVAE tokenizer from scratch")
    print("=" * 60)

    model = VQVAE(
        in_channels=3,
        hidden_dim=args.hidden_dim_vq,
        latent_dim=args.latent_dim,
        codebook_size=args.codebook_size,
        downsample_factor=args.downsample_factor,
    ).to(device)

    optimizer = AdamW(
        [p for n, p in model.named_parameters() if "perceptual_net" not in n],
        lr=args.lr_vqvae,
    )

    model.train()
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs_vqvae):
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            recon, indices, loss_dict = model(images)
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                used_codes = indices.unique().numel()
                print(
                    f"[VQVAE] epoch {epoch+1}/{args.epochs_vqvae} step {step} "
                    f"loss={loss_dict['loss'].item():.4f} "
                    f"mse={loss_dict['l_mse'].item():.4f} "
                    f"perc={loss_dict['l_p'].item():.4f} "
                    f"vq={loss_dict['l_q'].item():.4f} "
                    f"codes_used={used_codes}/{args.codebook_size} "
                    f"({elapsed:.0f}s)"
                )
            step += 1

    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, "vqvae_mini.pt")
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"Đã lưu VQVAE checkpoint: {ckpt_path}")
    return model


@torch.no_grad()
def encode_dataset_to_tokens(vqvae, loader, device):
    """Encode toàn bộ dataset thành (class_id, token_sequence) để train GPT.
    Tránh phải encode lại mỗi epoch -> nhanh hơn nhiều cho train GPT."""
    vqvae.eval()
    all_tokens, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        indices, _, _ = vqvae.encode(images)        # (B, h, w)
        B = indices.shape[0]
        flat = indices.reshape(B, -1)                # (B, N)
        all_tokens.append(flat.cpu())
        all_labels.append(labels)
    tokens = torch.cat(all_tokens, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return tokens, labels


def train_gpt(args, vqvae, train_loader, num_classes, device):
    print("\n" + "=" * 60)
    print("BƯỚC 2/2: Train GPT AR model from scratch (next-token prediction)")
    print("=" * 60)

    print("Encoding toàn bộ dataset train thành token (1 lần, dùng VQVAE đã đóng băng)...")
    tokens, labels = encode_dataset_to_tokens(vqvae, train_loader, device)
    seq_len = tokens.shape[1]
    print(f"Tokens shape: {tokens.shape} (N={seq_len} token/ảnh), labels shape: {labels.shape}")

    model = GPT_Mini(
        vocab_size=args.codebook_size,
        num_classes=num_classes,
        seq_len=seq_len,
        dim=args.gpt_dim,
        depth=args.gpt_depth,
        n_heads=args.gpt_heads,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr_gpt)

    dataset = torch.utils.data.TensorDataset(tokens, labels)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    model.train()
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs_gpt):
        epoch_loss = 0.0
        n_batches = 0
        for tok_batch, label_batch in loader:
            tok_batch = tok_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            logits = model(label_batch, tok_batch)
            loss = ntp_loss(logits, tok_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                ppl = torch.exp(loss.detach()).item()
                print(
                    f"[GPT] epoch {epoch+1}/{args.epochs_gpt} step {step} "
                    f"loss={loss.item():.4f} ppl={ppl:.2f} ({elapsed:.0f}s)"
                )
            step += 1

        print(f"==> Epoch {epoch+1} hoàn tất. Avg loss: {epoch_loss/n_batches:.4f}")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, "gpt_mini.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "seq_len": seq_len,
        "num_classes": num_classes,
    }, ckpt_path)
    print(f"Đã lưu GPT checkpoint: {ckpt_path}")
    return model


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Sử dụng device: {device}")
    print(f"Đáp ứng rubric 'Implement: Mini': train from scratch, "
          f"VQVAE {args.epochs_vqvae} epoch(s), GPT {args.epochs_gpt} epoch(s) "
          f"trên CIFAR-10 (>=1 epoch theo yêu cầu).")

    train_loader, test_loader, num_classes = get_cifar10_loaders(
        root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        train_subset=args.train_subset,
    )
    print(f"CIFAR-10: {len(train_loader.dataset)} ảnh train, {num_classes} class")

    vqvae = train_vqvae(args, train_loader, device)
    gpt = train_gpt(args, vqvae, train_loader, num_classes, device)

    print("\n" + "=" * 60)
    print("HOÀN TẤT Implement: Mini.")
    print(f"  - VQVAE checkpoint: {args.ckpt_dir}/vqvae_mini.pt")
    print(f"  - GPT checkpoint:   {args.ckpt_dir}/gpt_mini.pt")
    print("Hai checkpoint này sẽ được dùng làm điểm khởi đầu cho giai đoạn")
    print("Evaluation (VA-π RL fine-tuning trên >=2 dataset mới).")
    print("=" * 60)


if __name__ == "__main__":
    main()
