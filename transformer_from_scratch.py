import torch
from torch import nn
import math
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  SHARED: scaled dot-product attention
# ─────────────────────────────────────────────

def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)

    # (Q × K^T) / sqrt(d_k)  →  raw attention scores
    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(d_k)

    # mask hides future tokens in the decoder
    if mask is not None:
        scores = scores + mask

    # convert scores into probabilities
    attention_weights = F.softmax(scores, dim=-1)

    # weighted combination of values
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


# ─────────────────────────────────────────────
#  SHARED: Layer normalisation
# ─────────────────────────────────────────────

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps

        # learnable scale and shift parameters
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta  = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        dims = (-1,)

        mean     = x.mean(dim=dims, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=dims, keepdim=True)
        std      = torch.sqrt(variance + self.eps)

        normalized_x = (x - mean) / std
        return self.gamma * normalized_x + self.beta


# ─────────────────────────────────────────────
#  SHARED: Feed-forward network
#  Linear → ReLU → Dropout → Linear
# ─────────────────────────────────────────────

class FeedForwardNetwork(nn.Module):
    def __init__(self, model_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1       = nn.Linear(model_dim, hidden_dim)
        self.fc2       = nn.Linear(hidden_dim, model_dim)
        self.activation = nn.ReLU()
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ─────────────────────────────────────────────
#  ENCODER: Multi-head self-attention
# ─────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        assert model_dim % num_heads == 0, \
            "model_dim must be divisible by num_heads"

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim  = model_dim // num_heads

        # single projection creates Q, K, and V together
        self.qkv_projection    = nn.Linear(model_dim, 3 * model_dim)
        # combines all heads back together
        self.output_projection = nn.Linear(model_dim, model_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # create Q, K, V tensors
        qkv = self.qkv_projection(x)

        # split embeddings into multiple heads
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)

        # move heads dimension forward
        qkv = qkv.permute(0, 2, 1, 3)

        # separate query, key, value
        query, key, value = qkv.chunk(3, dim=-1)

        # attention calculation
        context, attention = scaled_dot_product_attention(query, key, value, mask)

        # combine all heads back
        context = context.reshape(batch_size, seq_len, self.model_dim)

        return self.output_projection(context)


# ─────────────────────────────────────────────
#  ENCODER: single block
# ─────────────────────────────────────────────

class TransformerEncoderBlock(nn.Module):
    def __init__(self, model_dim, hidden_dim, num_heads, dropout):
        super().__init__()

        self.self_attention = MultiHeadSelfAttention(model_dim, num_heads)
        self.norm1          = LayerNorm(model_dim)
        self.dropout1       = nn.Dropout(dropout)

        self.ffn            = FeedForwardNetwork(model_dim, hidden_dim, dropout)
        self.norm2          = LayerNorm(model_dim)
        self.dropout2       = nn.Dropout(dropout)

    def forward(self, x):
        # ---- Self-Attention block ----
        residual = x
        x = self.self_attention(x)
        x = self.dropout1(x)
        x = self.norm1(x + residual)

        # ---- Feed-Forward block ----
        residual = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual)

        return x


# ─────────────────────────────────────────────
#  ENCODER: full stack  (N layers)
# ─────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    def __init__(self, model_dim, hidden_dim, num_heads, dropout, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(model_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ─────────────────────────────────────────────
#  DECODER: Multi-head cross-attention
#  Q comes from the decoder, K/V from the encoder
# ─────────────────────────────────────────────

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        assert model_dim % num_heads == 0, \
            "model_dim must be divisible by num_heads"

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim  = model_dim // num_heads

        # keys and values come from encoder output
        self.kv_projection     = nn.Linear(model_dim, 2 * model_dim)
        # queries come from decoder input
        self.q_projection      = nn.Linear(model_dim, model_dim)
        self.output_projection = nn.Linear(model_dim, model_dim)

    def forward(self, encoder_output, decoder_input, mask=None):
        batch_size, seq_len, _ = encoder_output.size()

        # create keys and values from encoder output
        kv    = self.kv_projection(encoder_output)
        # create queries from decoder input
        query = self.q_projection(decoder_input)

        kv = kv.reshape(batch_size, seq_len, self.num_heads, 2 * self.head_dim)
        query = query.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        kv    = kv.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

        # split keys and values
        key, value = kv.chunk(2, dim=-1)

        # decoder attends to encoder output
        context, attention = scaled_dot_product_attention(query, key, value, mask)

        # merge all heads back
        context = context.reshape(batch_size, seq_len, self.model_dim)

        return self.output_projection(context)


# ─────────────────────────────────────────────
#  DECODER: single block
# ─────────────────────────────────────────────

class TransformerDecoderBlock(nn.Module):
    def __init__(self, model_dim, hidden_dim, num_heads, dropout):
        super().__init__()

        # masked self-attention (looks at previous tokens only)
        self.self_attention = MultiHeadSelfAttention(model_dim, num_heads)
        self.norm1          = LayerNorm(model_dim)
        self.dropout1       = nn.Dropout(dropout)

        # encoder-decoder cross-attention
        self.cross_attention = MultiHeadCrossAttention(model_dim, num_heads)
        self.norm2           = LayerNorm(model_dim)
        self.dropout2        = nn.Dropout(dropout)

        # feed-forward network
        self.ffn             = FeedForwardNetwork(model_dim, hidden_dim, dropout)
        self.norm3           = LayerNorm(model_dim)
        self.dropout3        = nn.Dropout(dropout)

    def forward(self, encoder_output, decoder_input, decoder_mask=None):
        # ---- Masked Self-Attention ----
        residual = decoder_input
        decoder_input = self.self_attention(decoder_input, mask=decoder_mask)
        decoder_input = self.dropout1(decoder_input)
        decoder_input = self.norm1(decoder_input + residual)

        # ---- Cross-Attention ----
        residual = decoder_input
        decoder_input = self.cross_attention(encoder_output, decoder_input)
        decoder_input = self.dropout2(decoder_input)
        decoder_input = self.norm2(decoder_input + residual)

        # ---- Feed-Forward Network ----
        residual = decoder_input
        decoder_input = self.ffn(decoder_input)
        decoder_input = self.dropout3(decoder_input)
        decoder_input = self.norm3(decoder_input + residual)

        return decoder_input


# ─────────────────────────────────────────────
#  DECODER: full stack  (N layers)
# ─────────────────────────────────────────────

class TransformerDecoder(nn.Module):
    def __init__(self, model_dim, hidden_dim, num_heads, dropout, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(model_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, encoder_output, decoder_input, decoder_mask=None):
        for layer in self.layers:
            decoder_input = layer(encoder_output, decoder_input, decoder_mask)
        return decoder_input


# ─────────────────────────────────────────────
#  TRANSFORMER: encoder + decoder + output head
# ─────────────────────────────────────────────

class Transformer(nn.Module):
    def __init__(
        self,
        model_dim,
        hidden_dim,
        num_heads,
        dropout,
        num_layers,
        vocab_size        # size of the target vocabulary
    ):
        super().__init__()

        # your encoder stack (from Encoder_transformer_from_scratch.ipynb)
        self.encoder = TransformerEncoder(
            model_dim, hidden_dim, num_heads, dropout, num_layers
        )

        # your decoder stack (from Decoder_Transformer_from_scratch.ipynb)
        self.decoder = TransformerDecoder(
            model_dim, hidden_dim, num_heads, dropout, num_layers
        )

        # project decoder output → vocabulary logits
        self.output_projection = nn.Linear(model_dim, vocab_size)

    def forward(
        self,
        src,                  # (batch, src_seq_len, model_dim)  encoder input
        tgt,                  # (batch, tgt_seq_len, model_dim)  decoder input
        decoder_mask=None     # causal mask, shape (tgt_seq_len, tgt_seq_len)
    ):
        # 1. Encode the source sequence
        encoder_output = self.encoder(src)           # (batch, src_seq, model_dim)

        # 2. Decode, cross-attending to encoder_output
        decoder_output = self.decoder(
            encoder_output,                          # K / V for cross-attention
            tgt,                                     # Q (target tokens so far)
            decoder_mask                             # hides future target positions
        )                                            # (batch, tgt_seq, model_dim)

        # 3. Project to vocabulary  (apply softmax externally, e.g. via CrossEntropyLoss)
        logits = self.output_projection(decoder_output)  # (batch, tgt_seq, vocab_size)
        return logits


# ─────────────────────────────────────────────
#  QUICK SMOKE-TEST  (same hyper-params as your notebooks)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("[debug] __main__ start")
    d_model    = 512
    ffn_hidden = 2048
    num_heads  = 8
    drop_prob  = 0.1
    num_layers = 5
    batch_size = 30
    src_len    = 200   # source sequence length  (e.g. English)
    tgt_len    = 200   # target sequence length  (e.g. Bangla)
    vocab_size = 10_000

    # causal mask: upper-triangle of -inf  (same mask you built in your decoder notebook)
    mask = torch.full([tgt_len, tgt_len], float('-inf'))
    mask = torch.triu(mask, diagonal=1)

    model = Transformer(
        model_dim  = d_model,
        hidden_dim = ffn_hidden,
        num_heads  = num_heads,
        dropout    = drop_prob,
        num_layers = num_layers,
        vocab_size = vocab_size,
    )

    # positional-encoded inputs  (same random tensors you used in both notebooks)
    src = torch.randn(batch_size, src_len, d_model)   # encoder input
    tgt = torch.randn(batch_size, tgt_len, d_model)   # decoder input

    print("[debug] before forward")
    logits = model(src, tgt, decoder_mask=mask)
    print("[debug] after forward")
    print("Output shape:", logits.shape)
    # → Output shape: torch.Size([30, 200, 10000])