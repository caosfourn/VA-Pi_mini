"""
Corruption kernel cho VA-π.

Cơ chế cốt lõi của VA-π (theo paper, Sec 3.2-3.3): để tạo ra sự đa dạng
cần thiết cho RL exploration MÀ KHÔNG CẦN free-running generation (vốn
tốn kém vì phải sinh tuần tự token-by-token), VA-π:

    1. Lấy ground-truth tokens x* (từ ảnh thật, qua tokenizer)
    2. Làm NHIỄU một phần context (thay một số token bằng token khác/random)
       -> tạo ra "noisy prefix" x~*
    3. Cho AR model chạy TEACHER FORCING trên noisy prefix này (1 forward
       pass, không cần generate tuần tự) để lấy logits tại mọi vị trí
    4. SAMPLE token mới từ các logits đó -> đây là "action" trong RL
    5. Decode token mới sample được thành ảnh -> tính reward pixel-space
       so với ảnh gốc thật

Điểm mấu chốt: việc làm nhiễu context (corruption) là thứ tạo ra "rollout"
đa dạng cho RL, thay thế cho việc phải free-running sample nhiều lần
(cách làm tốn kém của AR-GRPO truyền thống). Đây là lý do paper báo cáo
chỉ cần ~13.4% compute so với free-running RL.

Mức độ nhiễu được kiểm soát bởi corruption ratio xi (ξ) — tỷ lệ token
trong context bị thay thế. Paper dùng xi=0.5 (Table 7 - Hyperparameters).
"""

import torch


def corrupt_tokens(tokens: torch.Tensor, vocab_size: int, corrupt_ratio: float = 0.5,
                    generator: torch.Generator = None) -> torch.Tensor:
    """
    Làm nhiễu một tỷ lệ ngẫu nhiên token trong sequence bằng cách thay
    chúng bằng token ngẫu nhiên khác trong vocab (uniform random token).

    Đây tương ứng với corruption kernel q(x~* | x*) trong paper — bước
    đầu tiên để tạo "noisy context" trước khi teacher-forced sampling.

    tokens: (B, N) long tensor — ground truth token sequence x*
    vocab_size: kích thước codebook (số token khả dĩ)
    corrupt_ratio: xi (ξ) — tỷ lệ vị trí bị làm nhiễu, mặc định 0.5 theo Table 7

    Trả về: (B, N) tokens đã làm nhiễu (noisy prefix x~*), cùng mask vị trí
    nào đã bị thay đổi (hữu ích để phân tích/debug).
    """
    B, N = tokens.shape
    device = tokens.device

    mask = torch.rand(B, N, device=device, generator=generator) < corrupt_ratio
    random_tokens = torch.randint(
        0, vocab_size, (B, N), device=device, generator=generator
    )
    corrupted = torch.where(mask, random_tokens, tokens)
    return corrupted, mask


def corrupt_tokens_with_noise_schedule(tokens: torch.Tensor, vocab_size: int,
                                        min_ratio: float = 0.1, max_ratio: float = 0.5,
                                        generator: torch.Generator = None):
    """
    Biến thể: mỗi sample trong batch có một corrupt_ratio NGẪU NHIÊN riêng
    (lấy đều trong [min_ratio, max_ratio]) thay vì cố định cho cả batch.
    Giúp đa dạng hoá mức độ nhiễu giữa các rollout trong cùng 1 group GRPO,
    tăng exploration. Có thể dùng thay cho corrupt_tokens() nếu muốn nhiều
    đa dạng hơn trong cùng 1 group.
    """
    B, N = tokens.shape
    device = tokens.device

    ratios = torch.empty(B, 1, device=device).uniform_(min_ratio, max_ratio, generator=generator)
    rand_vals = torch.rand(B, N, device=device, generator=generator)
    mask = rand_vals < ratios  # broadcast (B,1) so với (B,N)

    random_tokens = torch.randint(0, vocab_size, (B, N), device=device, generator=generator)
    corrupted = torch.where(mask, random_tokens, tokens)
    return corrupted, mask
