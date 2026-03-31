import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from model.config import SLMConfig


# LayerNorm Inplementation (without bias) - as used in GPT-2
class LayerNorm(nn.Module):
    """Layer normalization applied across the embedding dimension."""
    
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(config.d_emb))
        self.shift = nn.Parameter(torch.zeros(config.d_emb))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize across embedding dimension
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normalized_x = (x - mean) / (std + self.eps)
        return self.scale.to(x.device) * normalized_x + self.shift.to(x.device)


class GELU(nn.Module):
    """Gaussian Error Linear Unit activation."""

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)
        ))

# Feed Forward Network 

class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, config: SLMConfig):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(config.d_emb, 4 * config.d_emb),
            GELU(),
            nn.Linear(4 * config.d_emb, config.d_emb)
        )

    def forward(self, x):
        return self.layers(x)


# Causal Multi-Head Attention  
class CausalMultiHeadAttention(nn.Module):
    """
    Multi-head causal self-attention
    """

    def __init__(self, config: SLMConfig):
        super().__init__()

        self.n_heads = config.n_heads
        self.head_dim = config.head_d_emb
        self.d_emb = config.d_emb

        # Q K V projections
        self.W_query = nn.Linear(config.d_emb, config.d_emb, bias=config.qkv_bias)
        self.W_key = nn.Linear(config.d_emb, config.d_emb, bias=config.qkv_bias)
        self.W_value = nn.Linear(config.d_emb, config.d_emb, bias=config.qkv_bias)

        # output projection
        self.out_proj = nn.Linear(config.d_emb, config.d_emb)

    def forward(self, x):

        batch, tokens, d_emb = x.shape

        # create Q K V
        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)

        return Q, K, V


# Causal Self-Attention Block
class CausalMultiHeadAttention(nn.Module):
    """
    Multi-head causal self-attention with masking
    """

    def __init__(self, config: SLMConfig):
        super().__init__()

        self.n_heads = config.n_heads
        self.head_dim = config.head_d_emb
        self.d_emb = config.d_emb

        # Q K V projections
        self.W_query = nn.Linear(config.d_emb, config.d_emb, bias=config.qkv_bias)
        self.W_key = nn.Linear(config.d_emb, config.d_emb, bias=config.qkv_bias)
        self.W_value = nn.Linear(config.d_emb, config.d_emb, bias=config.qkv_bias)

        # Output projection
        self.out_proj = nn.Linear(config.d_emb, config.d_emb)

        # Dropout
        self.attn_dropout = nn.Dropout(config.drop_rate)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(config.n_blocks, config.n_blocks), diagonal=1)
        )

    def forward(self, x):

        batch, tokens, _ = x.shape

        # Q K V projections
        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)

        # Split into heads
        Q = Q.view(batch, tokens, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, tokens, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, tokens, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)

        # Applying causal mask
        mask = self.mask[:tokens, :tokens].bool()
        scores = scores.masked_fill(mask, float('-inf'))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Context vector
        context = attn_weights @ V

        # Merge heads
        context = context.transpose(1, 2).contiguous().view(batch, tokens, self.d_emb)

        # Final projection
        output = self.out_proj(context)

        return output
    
# Transformer Block
class TransformerBlock(nn.Module):
    """Single transformer decoder block"""

    def __init__(self, config: SLMConfig):
        super().__init__()

        self.norm1 = LayerNorm(config)
        self.attention = CausalMultiHeadAttention(config)

        self.norm2 = LayerNorm(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):

        # Attention + residual
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = x + shortcut

        # FeedForward + residual
        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = x + shortcut

        return x


class GPT(nn.Module):
    """Complete GPT decoder model."""
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_emb)
        # Position embedding (using n_blocks from your config)
        self.position_embedding = nn.Embedding(config.n_blocks, config.d_emb)

        # Transformer blocks
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layers)]
        )
        
        self.final_norm = LayerNorm(config)

        # Output head
        self.output_head = nn.Linear(config.d_emb, config.vocab_size, bias=False)

        # Weight tying
        self.token_embedding.weight = self.output_head.weight

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.shape

        # Check sequence length against n_blocks
        assert t <= self.config.n_blocks, f"Sequence length {t} exceeds context {self.config.n_blocks}"

        tok_emb = self.token_embedding(idx)
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.output_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.n_blocks else idx[:, -self.config.n_blocks:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = torch.softmax(logits, dim=-1)

            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_probs[cumulative_probs > top_p] = 0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(sorted_probs, 1)
                idx_next = sorted_indices.gather(-1, next_token)
            else:
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx