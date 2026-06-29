"""
GRPO (Group Relative Policy Optimization) cho VA-π.

Theo paper (Sec 3.3, công thức tương ứng Eq. 9, 11):

1. Group-relative advantage:
   Với mỗi điều kiện (ảnh/class), sample G rollout (G ứng viên token sequence
   khác nhau nhờ corruption ngẫu nhiên ở các mức khác nhau). Mỗi rollout có
   reward r_i (Eq. 10). Advantage được tính CHUẨN HÓA TRONG GROUP:

       A_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + eps)

   Việc chuẩn hoá trong group giúp giảm phương sai, không cần value network
   riêng (khác PPO truyền thống).

2. GRPO objective có 2 phần:
   a) Clipped policy-ratio objective (giống PPO-clip):
        rho_i = pi_theta(o_i|q) / pi_theta_old(o_i|q)
        L_clip = -E[ min(rho_i * A_i, clip(rho_i, 1-eps, 1+eps) * A_i) ]

   b) KL-regularization để không lệch quá xa policy gốc (tham chiếu):
        L_kl = beta * KL(pi_theta || pi_ref)

   Điểm khác biệt của VA-π so với AR-GRPO chuẩn: KHÔNG cần giữ một
   reference model riêng để rollout thêm — vì cả phần policy-ratio
   VÀ phần regularization đều được tính trực tiếp từ CÙNG MỘT
   teacher-forcing forward pass (trên noisy context), không cần thêm
   lần generate/rollout nào => tiết kiệm rất nhiều compute (paper báo
   cáo chỉ cần ~13.4% compute so với free-running RL).

3. NTP regularization (giữ năng lực sinh ảnh gốc, tránh "reward hacking"
   làm model quên next-token-prediction thông thường):
        L_ntp = CrossEntropy(logits_on_clean_context, ground_truth_tokens)

   Tổng loss huấn luyện:
        L = L_clip + beta * L_kl  +  lambda_ntp * L_ntp
"""

import torch
import torch.nn.functional as F


def compute_group_relative_advantage(rewards: torch.Tensor, group_size: int,
                                      eps: float = 1e-6) -> torch.Tensor:
    """
    rewards: (B,) với B = num_conditions * group_size, được sắp xếp liên tiếp
             theo từng group (rewards[0:G] là group 1, rewards[G:2G] là group 2, ...)
    group_size: G — số rollout/sample mỗi condition.

    Trả về advantage (B,) đã chuẩn hoá trong group (mean=0, std=1 mỗi group).
    """
    B = rewards.shape[0]
    assert B % group_size == 0, f"Batch size {B} phải chia hết cho group_size {group_size}"
    num_groups = B // group_size

    r = rewards.reshape(num_groups, group_size)
    mean = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, keepdim=True)
    advantage = (r - mean) / (std + eps)
    return advantage.reshape(B)


def sequence_log_prob(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """
    Tính log pi(tokens | context) TỔNG trên cả sequence (sum log-prob từng
    vị trí) — dùng để tính policy ratio rho = pi_theta/pi_theta_old.

    logits: (B, N, vocab_size)
    tokens: (B, N) token đã được sample (action)
    Trả về: (B,) tổng log-probability của cả sequence.
    """
    log_probs = F.log_softmax(logits, dim=-1)  # (B, N, V)
    token_log_probs = torch.gather(
        log_probs, dim=-1, index=tokens.unsqueeze(-1)
    ).squeeze(-1)  # (B, N)
    return token_log_probs.sum(dim=1)  # (B,)


def grpo_clipped_objective(new_logits: torch.Tensor, old_logits: torch.Tensor,
                            sampled_tokens: torch.Tensor, advantage: torch.Tensor,
                            clip_eps: float = 0.2) -> torch.Tensor:
    """
    Phần (a) của GRPO: clipped policy-ratio objective.

    new_logits: (B, N, V) — logits từ policy hiện tại (pi_theta), forward
                lại trên CÙNG noisy context để có gradient.
    old_logits: (B, N, V) — logits từ policy TRƯỚC khi update (pi_theta_old),
                lấy từ lúc sample hành động, KHÔNG có gradient (detach).
    sampled_tokens: (B, N) — action đã sample (x_sample trong Eq. 10)
    advantage: (B,) — group-relative advantage đã chuẩn hoá

    Trả về: scalar loss (đã lấy trung bình, dấu âm vì ta MINIMIZE loss
    nhưng GRPO objective gốc là MAXIMIZE).
    """
    new_log_prob = sequence_log_prob(new_logits, sampled_tokens)
    old_log_prob = sequence_log_prob(old_logits, sampled_tokens).detach()

    log_ratio = new_log_prob - old_log_prob
    ratio = torch.exp(log_ratio)

    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage

    loss = -torch.min(unclipped, clipped).mean()
    return loss


def kl_penalty(new_logits: torch.Tensor, ref_logits: torch.Tensor,
               sampled_tokens: torch.Tensor) -> torch.Tensor:
    """
    Phần (b) của GRPO: ước lượng KL(pi_theta || pi_ref) qua k3 estimator
    (Schulman's approximation), không âm, phương sai thấp:
        KL_hat = exp(log_ref - log_new) - (log_ref - log_new) - 1

    ref_logits: logits từ policy tham chiếu (checkpoint Giai đoạn 2, GIỮ CỐ ĐỊNH),
                tính trên CÙNG noisy context trong CÙNG 1 lần forward.
    """
    new_log_prob = sequence_log_prob(new_logits, sampled_tokens)
    ref_log_prob = sequence_log_prob(ref_logits, sampled_tokens).detach()

    log_ratio = ref_log_prob - new_log_prob
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    return kl.mean()


def ntp_regularization_loss(clean_logits: torch.Tensor, ground_truth_tokens: torch.Tensor) -> torch.Tensor:
    """
    L_ntp: cross-entropy thường giữa logits (tính trên context SẠCH, không
    nhiễu) và ground-truth token — giữ lại khả năng next-token-prediction
    gốc, tránh model "quên" cách sinh ảnh bình thường chỉ vì tối ưu reward.
    """
    B, N, V = clean_logits.shape
    return F.cross_entropy(clean_logits.reshape(-1, V), ground_truth_tokens.reshape(-1))


def vapi_total_loss(clip_loss: torch.Tensor, kl_loss: torch.Tensor, ntp_loss: torch.Tensor,
                     beta: float = 0.1, lambda_ntp: float = 1.0) -> torch.Tensor:
    """
    Tổng hợp loss cuối cùng dùng để backward:
        L = L_clip + beta * L_kl + lambda_ntp * L_ntp
    beta: hệ số KL-regularization — paper dùng beta=0.1 (Table 7)
    """
    return clip_loss + beta * kl_loss + lambda_ntp * ntp_loss
