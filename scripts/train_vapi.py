"""
Script VA-π RL fine-tuning (Giai đoạn 3 — Evaluation).

Đáp ứng yêu cầu rubric:
    "Evaluation: Fine-tuning the trained model (Implement=Mini) for testing
    on at least 2 novel dataset"

Pipeline 1 step GRPO (lặp lại nhiều step):
    1. Lấy batch ảnh thật + class label từ dataset MỚI (CIFAR-100/STL-10)
    2. Encode ảnh -> ground-truth tokens x* (qua VQVAE đã đóng băng từ Giai đoạn 2)
    3. Với mỗi (ảnh, class), tạo G "rollout" bằng cách:
         a. Corrupt context (Eq. corruption kernel, models/corruption.py)
            với G mức nhiễu khác nhau (group cho GRPO)
         b. Teacher-forced sample token mới từ noisy context (Eq. 7)
            -> đây vừa là "action", vừa cho ta old_logits (pi_theta_old)
            và ref_logits (pi_theta_ref, từ model GỐC giữ cố định)
         c. Decode token sample thành ảnh -> tính reward pixel-space (Eq. 10)
    4. Forward LẠI policy hiện tại (pi_theta, có gradient) trên CÙNG noisy
       context để lấy new_logits
    5. Tính GRPO loss = clip_loss + beta*kl_loss + lambda_ntp*ntp_loss
    6. Backward + optimizer step CHỈ trên GPT (VQVAE giữ đóng băng suốt
       quá trình, đúng tinh thần paper: "we only update the AR generator
       while keeping the tokenizer frozen")

Mỗi dataset mới (CIFAR-100, STL-10) được fine-tune RIÊNG, tạo ra 2
checkpoint khác nhau — đúng yêu cầu "testing on at least 2 novel dataset".

Cách chạy (trên Kaggle):
    python train_vapi.py --dataset cifar100 --vqvae_ckpt .../vqvae_mini.pt \
        --gpt_ckpt .../gpt_mini.pt --steps 500
    python train_vapi.py --dataset stl10  --vqvae_ckpt .../vqvae_mini.pt \
        --gpt_ckpt .../gpt_mini.pt --steps 500
"""

import argparse
import copy
import os
import sys
import time

import torch
from torch.optim import AdamW

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vqvae import VQVAE
from models.gpt import GPT_Mini
from models.corruption import corrupt_tokens_with_noise_schedule
from models.reward import pixel_reconstruction_reward
from models.grpo import (
    compute_group_relative_advantage,
    grpo_clipped_objective,
    kl_penalty,
    ntp_regularization_loss,
    vapi_total_loss,
)
from data.data_utils import get_cifar100_loaders, get_stl10_loaders


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vqvae_ckpt", type=str, required=True)
    p.add_argument("--gpt_ckpt", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True, choices=["cifar100", "stl10"],
                    help="Dataset MỚI (novel) dùng để fine-tune + evaluate.")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--ckpt_dir", type=str, default="./checkpoints")

    # GRPO / VA-π hyperparameters
    p.add_argument("--group_size", type=int, default=8, help="G trong GRPO (Table 7: G=8)")
    p.add_argument("--num_conditions_per_batch", type=int, default=8,
                    help="Số ảnh khác nhau mỗi step; batch thực tế = num_conditions * group_size")
    p.add_argument("--corrupt_min_ratio", type=float, default=0.1)
    p.add_argument("--corrupt_max_ratio", type=float, default=0.5,
                    help="xi (ξ) tối đa — Table 7 dùng xi=0.5")
    p.add_argument("--beta", type=float, default=0.1, help="KL-regularization coeff (Table 7: beta=0.1)")
    p.add_argument("--lambda_ntp", type=float, default=1.0, help="Hệ số NTP regularization")
    p.add_argument("--clip_eps", type=float, default=0.2, help="PPO/GRPO clip epsilon")
    p.add_argument("--reward_mode", type=str, default="gaussian", choices=["gaussian", "neg_mse"])
    p.add_argument("--reward_tau", type=float, default=0.5)

    p.add_argument("--steps", type=int, default=500,
                    help="Số GRPO update step (= số lần lấy rollout MỚI).")
    p.add_argument("--inner_epochs", type=int, default=4,
                    help="Số lần gradient update TRÊN CÙNG 1 rollout trước khi lấy rollout mới "
                         "(giống PPO 'ppo_epochs'). BẮT BUỘC > 1, nếu không clip_loss/kl_loss "
                         "sẽ luôn ≈0 vì new_logits trùng old_logits ngay sau khi sample.")
    p.add_argument("--lr", type=float, default=1e-6, help="Learning rate (Table 7: lr=1e-6)")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_frozen_vqvae(vqvae_ckpt_path, device):
    ckpt = torch.load(vqvae_ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    vqvae = VQVAE(
        in_channels=3,
        hidden_dim=args["hidden_dim_vq"],
        latent_dim=args["latent_dim"],
        codebook_size=args["codebook_size"],
        downsample_factor=args["downsample_factor"],
    ).to(device)
    vqvae.load_state_dict(ckpt["model_state_dict"])
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)  # ĐÓNG BĂNG tokenizer — đúng tinh thần VA-π
    return vqvae, args


def load_gpt_policy(gpt_ckpt_path, device):
    ckpt = torch.load(gpt_ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    gpt = GPT_Mini(
        vocab_size=args["codebook_size"],
        num_classes=ckpt["num_classes"],
        seq_len=ckpt["seq_len"],
        dim=args["gpt_dim"],
        depth=args["gpt_depth"],
        n_heads=args["gpt_heads"],
    ).to(device)
    gpt.load_state_dict(ckpt["model_state_dict"])
    return gpt, ckpt["seq_len"], ckpt["num_classes"], args


def get_novel_loader(args):
    """Load dataset mới. LƯU Ý quan trọng: dataset mới (CIFAR-100/STL-10)
    có SỐ CLASS KHÁC với CIFAR-10 (10 class lúc train GPT). Vì model C2I
    cần class_id nằm trong [0, num_classes_train), ta MAP class mới về
    không gian class cũ bằng modulo — giả định đơn giản hoá hợp lý cho
    Mini (ghi rõ trong báo cáo): coi class mới là "điều kiện lạ" được áp
    vào không gian điều kiện đã học, không mở rộng embedding mới."""
    if args.dataset == "cifar100":
        loader, num_classes = get_cifar100_loaders(
            root=args.data_root, image_size=32, batch_size=64
        )
    else:
        loader, num_classes = get_stl10_loaders(
            root=args.data_root, image_size=32, batch_size=64
        )
    return loader, num_classes


def remap_labels(labels, num_classes_novel, num_classes_train):
    """Map nhãn dataset mới về không gian class mà GPT đã học, bằng modulo."""
    return labels % num_classes_train


def sample_one_grpo_group(gpt, vqvae, class_ids, gt_tokens, args, device):
    """
    Với MỘT batch điều kiện (class_ids, gt_tokens), mở rộng thành group_size
    rollout mỗi điều kiện, teacher-forced sample action + reward.
    """
    G = args.group_size

    class_ids_rep = class_ids.repeat_interleave(G, dim=0)
    gt_tokens_rep = gt_tokens.repeat_interleave(G, dim=0)

    noisy_tokens, _ = corrupt_tokens_with_noise_schedule(
        gt_tokens_rep, vocab_size=vqvae.codebook_size,
        min_ratio=args.corrupt_min_ratio, max_ratio=args.corrupt_max_ratio,
    )

    with torch.no_grad():
        old_logits = gpt(class_ids_rep, noisy_tokens)
        probs = torch.softmax(old_logits, dim=-1)
        B, N, V = probs.shape
        sampled_tokens = torch.multinomial(probs.reshape(-1, V), num_samples=1).reshape(B, N)

        side = int(round(N ** 0.5))
        gen_images = vqvae.decode_from_indices(sampled_tokens.reshape(B, side, side))

    return {
        "class_ids_rep": class_ids_rep,
        "gt_tokens_rep": gt_tokens_rep,
        "noisy_tokens": noisy_tokens,
        "old_logits": old_logits,
        "sampled_tokens": sampled_tokens,
        "gen_images": gen_images,
    }


def train_vapi(args):
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"=== VA-π RL fine-tuning trên dataset MỚI: {args.dataset} ===")

    vqvae, vq_args = load_frozen_vqvae(args.vqvae_ckpt, device)
    gpt, seq_len, num_classes_train, gpt_args = load_gpt_policy(args.gpt_ckpt, device)
    print(f"Đã load VQVAE (frozen) và GPT policy. seq_len={seq_len}, "
          f"num_classes (lúc train Mini)={num_classes_train}")

    # reference policy: BẢN SAO CỐ ĐỊNH của GPT tại thời điểm bắt đầu Giai đoạn 3.
    ref_gpt = copy.deepcopy(gpt).to(device)
    ref_gpt.eval()
    for p in ref_gpt.parameters():
        p.requires_grad_(False)

    loader, num_classes_novel = get_novel_loader(args)
    print(f"Dataset mới '{args.dataset}': {len(loader.dataset)} ảnh, "
          f"{num_classes_novel} class (map về {num_classes_train} class đã train qua modulo)")

    optimizer = AdamW(gpt.parameters(), lr=args.lr)
    data_iter = iter(loader)

    def next_batch():
        nonlocal data_iter
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, labels = next(data_iter)
        return images, labels

    gpt.train()
    t0 = time.time()
    history = []

    for step in range(1, args.steps + 1):
        images, labels = next_batch()
        images = images[: args.num_conditions_per_batch].to(device)
        labels = labels[: args.num_conditions_per_batch]
        labels = remap_labels(labels, num_classes_novel, num_classes_train).to(device)

        with torch.no_grad():
            gt_indices, _, _ = vqvae.encode(images)
            gt_tokens = gt_indices.reshape(images.shape[0], -1)

        # --- 1 lần rollout (sample action + reward), dùng args.group_size mức nhiễu khác nhau ---
        rollout = sample_one_grpo_group(gpt, vqvae, labels, gt_tokens, args, device)

        ref_images_rep = images.repeat_interleave(args.group_size, dim=0)
        rewards = pixel_reconstruction_reward(
            rollout["gen_images"], ref_images_rep,
            mode=args.reward_mode, tau=args.reward_tau,
        )
        advantage = compute_group_relative_advantage(rewards, group_size=args.group_size)

        # --- nhiều lần gradient update TRÊN CÙNG rollout này (giống PPO ppo_epochs) ---
        # Quan trọng: nếu inner_epochs=1, new_logits sẽ TRÙNG old_logits (vì cả hai forward
        # trên cùng tham số gpt chưa kịp đổi) => ratio=exp(0)=1 luôn => clip_loss≈0 luôn vì
        # advantage đã chuẩn hoá mean=0/group. Cần >=2 inner epoch để ratio thực sự có ý nghĩa.
        last_losses = None
        for inner_step in range(args.inner_epochs):
            new_logits = gpt(rollout["class_ids_rep"], rollout["noisy_tokens"])

            with torch.no_grad():
                ref_logits = ref_gpt(rollout["class_ids_rep"], rollout["noisy_tokens"])

            clip_loss = grpo_clipped_objective(
                new_logits, rollout["old_logits"], rollout["sampled_tokens"], advantage,
                clip_eps=args.clip_eps,
            )
            kl_loss = kl_penalty(new_logits, ref_logits, rollout["sampled_tokens"])

            clean_logits = gpt(rollout["class_ids_rep"], rollout["gt_tokens_rep"])
            ntp_loss = ntp_regularization_loss(clean_logits, rollout["gt_tokens_rep"])

            total_loss = vapi_total_loss(
                clip_loss, kl_loss, ntp_loss, beta=args.beta, lambda_ntp=args.lambda_ntp
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(gpt.parameters(), max_norm=1.0)
            optimizer.step()

            last_losses = (total_loss.item(), clip_loss.item(), kl_loss.item(), ntp_loss.item())

        total_l, clip_l, kl_l, ntp_l = last_losses
        history.append({
            "step": step,
            "total_loss": total_l,
            "clip_loss": clip_l,
            "kl_loss": kl_l,
            "ntp_loss": ntp_l,
            "mean_reward": rewards.mean().item(),
        })

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            print(
                f"[{args.dataset}] step {step}/{args.steps} "
                f"loss={total_l:.4f} clip={clip_l:.4f} "
                f"kl={kl_l:.4f} ntp={ntp_l:.4f} "
                f"reward={rewards.mean().item():.4f} ({elapsed:.0f}s)"
            )

        if step % args.save_every == 0 or step == args.steps:
            os.makedirs(args.ckpt_dir, exist_ok=True)
            save_path = os.path.join(args.ckpt_dir, f"gpt_vapi_{args.dataset}.pt")
            torch.save({
                "model_state_dict": gpt.state_dict(),
                "args": gpt_args,
                "seq_len": seq_len,
                "num_classes": num_classes_train,
                "vapi_train_args": vars(args),
                "step": step,
            }, save_path)

    import json
    history_path = os.path.join(args.ckpt_dir, f"vapi_history_{args.dataset}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nHOÀN TẤT VA-π fine-tuning trên '{args.dataset}'.")
    print(f"  - Checkpoint: {args.ckpt_dir}/gpt_vapi_{args.dataset}.pt")
    print(f"  - Lịch sử loss: {history_path}")


if __name__ == "__main__":
    args = parse_args()
    train_vapi(args)
