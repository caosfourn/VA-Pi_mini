"""
Autoregressive (AR) Transformer cho VA-π (Mini implementation).

Tương ứng Sec 3.1, công thức (3) của paper:
    theta = argmax_theta  sum_{i=1}^{N} log pi_theta(x_i | x_{1:i-1})

Đây là kiến trúc GPT decoder-only thu nhỏ (so với GPT-B/L/XL/XXL của
LlamaGen gốc) — đủ nhỏ để train from scratch trong vài giờ trên GPU
Kaggle (T4/P100), nhưng giữ đúng cấu trúc: causal self-attention,
class-conditioning token ở đầu sequence (giống LlamaGen C2I), và
next-token-prediction loss chuẩn.

Vai trò trong pipeline:
- Giai đoạn "Implement: Mini": model NÀY được train from scratch bằng
  NTP loss thường (Eq. 3) trên dataset nhỏ (CIFAR-10).
- Giai đoạn "Evaluation" (VA-π RL fine-tuning): model NÀY chính là
  policy pi_theta trong công thức GRPO (Eq. 11) — được fine-tune tiếp
  bằng reward pixel-space + regularizer NTP-noisy, KHÔNG train from scratch.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each: (B, n_heads, T, head_dim)

        # causal scaled-dot-product attention (PyTorch >=2.0 có sẵn flash attn)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT_Mini(nn.Module):
    """
    AR Transformer nhỏ cho class-conditional image generation (C2I).

    Sequence layout (giống LlamaGen):
        [class_token, image_token_1, image_token_2, ..., image_token_N]
    Class token nằm ở vị trí 0, đóng vai trò conditioning. Loss NTP chỉ
    tính trên các vị trí dự đoán image token (vị trí 1..N).
    """

    def __init__(
        self,
        vocab_size: int,          # = codebook_size của VQVAE
        num_classes: int,         # số class điều kiện (C2I)
        seq_len: int,             # N = số token ảnh (không tính class token)
        dim: int = 256,
        depth: int = 6,
        n_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.total_len = seq_len + 1  # +1 cho class-conditioning token

        # token embedding cho ảnh dùng chung không gian với class embedding
        # (class id được offset để không đụng vocab token ảnh)
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.cls_emb = nn.Embedding(num_classes, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.total_len, dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def _embed_sequence(self, class_ids, image_tokens):
        """
        class_ids: (B,) long
        image_tokens: (B, T) long, T <= seq_len  (token ảnh đã biết, dùng làm input/prefix)
        Trả về embedding sequence độ dài T+1 (gồm class token ở đầu).
        """
        B = class_ids.shape[0]
        cls_tok = self.cls_emb(class_ids).unsqueeze(1)        # (B, 1, dim)
        if image_tokens.shape[1] > 0:
            img_tok = self.tok_emb(image_tokens)               # (B, T, dim)
            x = torch.cat([cls_tok, img_tok], dim=1)            # (B, T+1, dim)
        else:
            x = cls_tok
        x = x + self.pos_emb[:, : x.shape[1], :]
        return self.drop(x)

    def forward(self, class_ids, image_tokens):
        """
        Teacher-forcing forward pass dùng cho training (NTP loss, Eq. 3/9).

        class_ids: (B,)
        image_tokens: (B, N) ground-truth (hoặc đã làm nhiễu) token ảnh, N = seq_len

        Trả về logits (B, N, vocab_size) — logits[:, i, :] dự đoán token thứ i
        dựa trên class_id + image_tokens[:, :i] (causal, đúng định nghĩa AR).
        """
        B, N = image_tokens.shape
        assert N == self.seq_len, f"Kỳ vọng {self.seq_len} token ảnh, nhận {N}"

        # input cho transformer: [class, tok_0, ..., tok_{N-2}]  (bỏ token cuối,
        # vì ta cần dự đoán nó, không cần đưa vào input)
        input_tokens = image_tokens[:, :-1]  # (B, N-1)
        x = self._embed_sequence(class_ids, input_tokens)  # (B, N, dim)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, N, vocab_size) — vị trí i dự đoán image_tokens[:, i]
        return logits

    @torch.no_grad()
    def generate(self, class_ids, temperature=1.0, top_k=None, top_p=None):
        """
        Free-running sampling (inference thường, KHÔNG dùng teacher forcing).
        Dùng để sinh ảnh mới từ class label, hoặc để quan sát chất lượng
        model trước/sau VA-π fine-tuning.
        """
        B = class_ids.shape[0]
        device = class_ids.device
        generated = torch.zeros(B, 0, dtype=torch.long, device=device)

        for _ in range(self.seq_len):
            x = self._embed_sequence(class_ids, generated)
            for block in self.blocks:
                x = block(x)
            x = self.ln_f(x)
            logits = self.head(x)[:, -1, :] / max(temperature, 1e-5)  # (B, vocab)

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(probs, dim=-1)
                mask = cum_probs - probs > top_p
                sorted_logits[mask] = -float("inf")
                logits = torch.full_like(logits, -float("inf")).scatter(
                    1, sorted_idx, sorted_logits
                )

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated  # (B, seq_len)

    @torch.no_grad()
    def teacher_forced_sample(self, class_ids, context_tokens, temperature=1.0):
        """
        Sample tokens bằng teacher forcing (Eq. 7 trong paper):
            q(x|I) = prod_i pi_theta(x_i | x*_{1:i-1})
        Mỗi token được SAMPLE (không phải argmax) từ phân phối điều kiện trên
        PREFIX GỐC (có thể đã làm nhiễu), không phải trên token tự sinh trước đó.
        Đây chính là cách lấy "x" trong reward reconstruction (Eq. 10) và là
        điểm khác biệt mấu chốt giữa VA-π và free-running RL thông thường
        (không cần generate tự hồi quy tốn kém).

        context_tokens: (B, N) prefix dùng làm điều kiện (x* hoặc x~* đã nhiễu)
        Trả về (B, N) — token được sample tại mỗi vị trí dựa trên prefix context_tokens.
        """
        logits = self.forward(class_ids, context_tokens)  # (B, N, vocab) — full teacher-forced
        logits = logits / max(temperature, 1e-5)
        probs = F.softmax(logits, dim=-1)
        B, N, V = probs.shape
        sampled = torch.multinomial(probs.reshape(-1, V), num_samples=1).reshape(B, N)
        return sampled, logits


def ntp_loss(logits, targets):
    """
    Next-token-prediction loss chuẩn (Eq. 3, dạng cross-entropy):
        loss = - (1/N) * sum_i log pi_theta(x_i | x_{<i})

    logits: (B, N, vocab_size)
    targets: (B, N) ground-truth token indices
    """
    B, N, V = logits.shape
    return F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
