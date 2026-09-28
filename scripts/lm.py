import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# RMSNorm
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


# ============================================================
# Rotary Positional Embedding
# ============================================================

def precompute_rope(head_dim, max_seq_len, theta=10000.0, device=None):
    """
    Precompute cosine and sine tables for RoPE.

    Returns:
        cos: [max_seq_len, head_dim]
        sin: [max_seq_len, head_dim]
    """

    assert head_dim % 2 == 0, "head_dim must be even"

    inv_freq = 1.0 / (
        theta ** (
            torch.arange(0, head_dim, 2, device=device).float()
            / head_dim
        )
    )

    positions = torch.arange(
        max_seq_len,
        device=device,
        dtype=torch.float32
    )

    freqs = torch.outer(positions, inv_freq)

    emb = torch.cat([freqs, freqs], dim=-1)

    cos = emb.cos()
    sin = emb.sin()

    return cos, sin


def rotate_half(x):
    """
    [x1, x2] -> [-x2, x1]
    """
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    """
    q, k:
        [B, heads, seq_len, head_dim]

    cos, sin:
        [seq_len, head_dim]
    """

    seq_len = q.shape[-2]

    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)

    return q, k


# ============================================================
# SwiGLU Feed Forward Network
# ============================================================

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(
            F.silu(self.w1(x)) * self.w2(x)
        )


# ============================================================
# Causal Self Attention
# ============================================================

class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        max_seq_len,
        rope_theta=10000.0
    ):
        super().__init__()

        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        cos, sin = precompute_rope(
            self.head_dim,
            max_seq_len,
            rope_theta
        )

        self.register_buffer(
            "rope_cos",
            cos,
            persistent=False
        )

        self.register_buffer(
            "rope_sin",
            sin,
            persistent=False
        )

    def forward(self, x):

        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        q, k = apply_rope(
            q,
            k,
            self.rope_cos,
            self.rope_sin
        )

        # PyTorch's fused scaled-dot-product attention.
        # is_causal=True creates the causal mask.
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True
        )

        y = y.transpose(1, 2).contiguous()

        y = y.view(B, T, C)

        return self.o_proj(y)


# ============================================================
# Transformer Block
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        ffn_hidden_dim,
        max_seq_len
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim)

        self.attention = CausalSelfAttention(
            dim=dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len
        )

        self.norm2 = RMSNorm(dim)

        self.ffn = SwiGLU(
            dim=dim,
            hidden_dim=ffn_hidden_dim
        )

    def forward(self, x):

        x = x + self.attention(
            self.norm1(x)
        )

        x = x + self.ffn(
            self.norm2(x)
        )

        return x


# ============================================================
# SkyVLM Language Model
# ============================================================

class SkyVLMForCausalLM(nn.Module):

    def __init__(
        self,
        vocab_size=16384,
        dim=512,
        num_layers=20,
        num_heads=8,
        ffn_hidden_dim=1408,
        max_seq_len=512
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(
            vocab_size,
            dim
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                ffn_hidden_dim=ffn_hidden_dim,
                max_seq_len=max_seq_len
            )
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(dim)

        self.lm_head = nn.Linear(
            dim,
            vocab_size,
            bias=False
        )

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(
        self,
        input_ids,
        labels=None
    ):

        B, T = input_ids.shape

        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"max_seq_len={self.max_seq_len}"
            )

        x = self.token_embedding(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        logits = self.lm_head(x)

        loss = None

        if labels is not None:

            loss = F.cross_entropy(
                logits.float().view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=50,
        temperature=1.0,
        top_k=None
    ):

        self.eval()

        for _ in range(max_new_tokens):

            idx_cond = input_ids[:, -self.max_seq_len:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]

            logits = logits / max(temperature, 1e-5)

            if top_k is not None:

                values, _ = torch.topk(
                    logits,
                    min(top_k, logits.size(-1))
                )

                logits[
                    logits < values[:, [-1]]
                ] = -float("inf")

            probs = F.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probs,
                num_samples=1
            )

            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1
            )

        return input_ids


# ============================================================
# Parameter Count
# ============================================================

def count_parameters(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SkyVLM Language Model")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    model = SkyVLMForCausalLM().to(device)

    total, trainable = count_parameters(model)

    print(
        f"Parameters: {total:,}"
    )

    print(
        f"Parameters (M): "
        f"{total / 1e6:.2f}M"
    )

    # Dummy batch
    batch_size = 2
    seq_len = 128

    input_ids = torch.randint(
        0,
        16384,
        (batch_size, seq_len),
        device=device
    )

    labels = torch.randint(
        0,
        16384,
        (batch_size, seq_len),
        device=device
    )

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=device.type == "cuda"
    ):

        logits, loss = model(
            input_ids,
            labels
        )

    print(
        f"Input shape : {input_ids.shape}"
    )

    print(
        f"Logits shape: {logits.shape}"
    )

    print(
        f"Loss        : {loss.item():.4f}"
    )

    print("=" * 70)
    print("LM TEST PASSED")
    print("=" * 70)