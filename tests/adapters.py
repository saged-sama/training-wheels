from typing import Any
from jaxtyping import Float, Int, Bool

from torch import Tensor
import therapml_cpp
import numpy as np
from therapml.training.sgd import SGD
from therapml.training.adam import AdamW
from therapml.training.nn_blocks import ReLU, GELU, SoftMax, Linear, SwiGLU
from therapml.training.loss import CrossEntropyLoss
from therapml.training.dropout import Dropout
from therapml.training.normalizers import LayerNorm, RMSNorm
from therapml.transformer import RoPE, SelfAttention, MultiHeadSelfAttention, MultiHeadSelfAttentionWithRope
from therapml.transformer.my_llama.library.layers import TransformerBlock

def run_tensor_multiply(arr1: Float[list, "b x y"], arr2: Float[list, "b y z"]) -> Float[list, "b x z"]:
    a = np.ascontiguousarray(arr1, dtype=np.float64)
    b = np.ascontiguousarray(arr2, dtype=np.float64)
    return therapml_cpp.run_tensor_multiply(a, b)


def run_tensor_dot(arr1: Float[list, "..."], arr2: Float[list, "..."], dim: int):
    result = therapml_cpp.run_tensor_dot(arr1, arr2, dim)
    if isinstance(result, np.ndarray) and result.shape == ():
        return result.item()
    return result


def get_sgd_cls() -> Any:
    return SGD


def get_adam_cls() -> Any:
    return AdamW


def run_relu(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    relu = ReLU()
    return relu(in_features)


def run_gelu(in_features: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    gelu = GELU()
    return gelu(in_features)


def run_softmax(in_features: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    softMax = SoftMax(dim=dim)
    return softMax(in_features)

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, "d_out d_in"],
    in_features: Float[Tensor, "... d_in"],
) -> Float[Tensor, "... d_out"]:
    custom_linear = Linear(d_in, d_out)
    custom_linear.load_state_dict({"weights": weights})
    return custom_linear(in_features)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    swilgu = SwiGLU(d_model=d_model, d_ff=d_ff, w1_weight=w1_weight, w2_weight=w2_weight, w3_weight=w3_weight)

    return swilgu(in_features)


def run_cross_entropy_loss(
    logits: Float[Tensor, "batch output_dim"], ground_truth: Float[Tensor, "batch output_dim"]
) -> Float[Tensor, ""]:
    cross_entropy_loss = CrossEntropyLoss()
    return cross_entropy_loss(logits, ground_truth)


def run_dropout(input: Float[Tensor, "..."], prob: float) -> Float[Tensor, "..."]:
    dropout = Dropout(prob)
    return dropout(input)


def run_layernorm(
    input: Float[Tensor, "batch ..."], gamma: Float[Tensor, "batch ..."], beta: Float[Tensor, "batch ..."]
) -> Float[Tensor, "batch ..."]:
    layer_norm = LayerNorm(gamma=gamma, beta=beta)
    return layer_norm(input)


def run_rmsnorm(input: Float[Tensor, "batch ..."], gamma: Float[Tensor, "batch ..."]) -> Float[Tensor, "batch ..."]:
    rms_norm = RMSNorm(gamma=gamma)
    return rms_norm(input)


def run_rope(
    embedding_dim: int,
    theta: float,
    context_len: int,
    input_embeddings: Float[Tensor, "batch ctx_len embedding_dim"],
    token_positions: Int[Tensor, "batch ctx_len"],
) -> Float[Tensor, "batch ctx_len embedding_dim"]:
    rope = RoPE(embedding_dim=embedding_dim, theta=theta, context_len=context_len)
    return rope(input_embeddings, token_positions)


def run_self_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... batch keys d_k"],
    V: Float[Tensor, "... batch values d_v"],
    mask: Bool[Tensor, "... batch queries keys"] | None = None,
) -> Float[Tensor, "... batch queries d_v"]:
    """
    Note the number of dimensions here can be greater than 3
    """
    self_attention = SelfAttention(K, V, mask)

    return self_attention(Q)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, "d_k d_in"],
    k_proj_weight: Float[Tensor, "d_k d_in"],
    v_proj_weight: Float[Tensor, "d_v d_in"],
    o_proj_weight: Float[Tensor, "d_model d_v"],
    in_features: Float[Tensor, "batch ctx_len d_in"],
) -> Float[Tensor, "batch ctx_len d_out"]:
    multi_self_att = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight, v_proj_weight=v_proj_weight, o_proj_weight=o_proj_weight)

    return multi_self_att(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    ctx_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, "d_k d_in"],
    k_proj_weight: Float[Tensor, "d_k d_in"],
    v_proj_weight: Float[Tensor, "d_v d_in"],
    o_proj_weight: Float[Tensor, "d_model d_v"],
    in_features: Float[Tensor, "batch ctx_len d_in"],
    token_positions: Int[Tensor, "batch ctx_len"],
) -> Float[Tensor, "batch ctx_len d_out"]:
    
    multi_head_with_rope = MultiHeadSelfAttentionWithRope(
        d_model=d_model,
        num_heads=num_heads,
        ctx_len=ctx_len,
        theta=theta,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight
    )

    return multi_head_with_rope(in_features, token_positions)

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    ctx_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Note this function should use RoPE.

    Args:
        d_model (int): The dimensionality of the Transformer block input.

        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.

        d_ff (int): Dimensionality of the feed-forward inner layer.

        ctx_len (int): Maximum sequence length of the input_tensor.

        theta (float): RoPE parameter.

        weights (dict[str, Tensor]):
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).

        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    import torch
    import torch.nn.functional as F
    import math
    
    batch_size, seq_len, _ = in_features.shape
    
    # Extract weights
    ln1_weight = weights["ln1.weight"]
    ln2_weight = weights["ln2.weight"]
    q_proj_weight = weights["attn.q_proj.weight"]
    k_proj_weight = weights["attn.k_proj.weight"]
    v_proj_weight = weights["attn.v_proj.weight"]
    o_proj_weight = weights["attn.output_proj.weight"]
    w1_weight = weights["ffn.w1.weight"]
    w2_weight = weights["ffn.w2.weight"]
    w3_weight = weights["ffn.w3.weight"]
    
    head_dim = d_model // num_heads
    
    # First LayerNorm (RMSNorm)
    x = in_features
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
    x_norm = (x / rms) * ln1_weight
    
    # Attention with RoPE
    # Project to Q, K, V
    Q = x_norm @ q_proj_weight.T  # (batch, seq_len, d_model)
    K = x_norm @ k_proj_weight.T  # (batch, seq_len, d_model)
    V = x_norm @ v_proj_weight.T  # (batch, seq_len, d_model)
    
    # Split heads
    Q = Q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
    K = K.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    V = V.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # Apply RoPE
    rope = RoPE(head_dim, theta, ctx_len)
    token_positions = torch.arange(seq_len, device=in_features.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    
    # Reshape for rope: (batch, num_heads, seq_len, head_dim) -> (batch*num_heads, seq_len, head_dim)
    B, H, S, D = Q.shape
    Q_rope = Q.reshape(B * H, S, D)
    K_rope = K.reshape(B * H, S, D)
    
    # Expand positions for all heads
    positions = token_positions.expand(B, -1).repeat_interleave(H, dim=0)
    
    Q_rope = rope(Q_rope, positions)
    K_rope = rope(K_rope, positions)
    
    # Reshape back
    Q = Q_rope.reshape(B, H, S, D)
    K = K_rope.reshape(B, H, S, D)
    
    # Scaled dot-product attention
    scale = math.sqrt(head_dim)
    scores = (Q @ K.transpose(-2, -1)) / scale
    
    # Causal mask
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool),
        diagonal=1
    )
    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    # Softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply attention to V
    attn_out = attention_weights @ V  # (batch, num_heads, seq_len, head_dim)
    
    # Merge heads
    attn_out = attn_out.transpose(1, 2).contiguous()  # (batch, seq_len, num_heads, head_dim)
    attn_out = attn_out.view(batch_size, seq_len, d_model)
    
    # Output projection
    attn_out = attn_out @ o_proj_weight.T
    
    # First residual
    x = in_features + attn_out
    
    # Second LayerNorm (RMSNorm)
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-8)
    x_norm = (x / rms) * ln2_weight
    
    # Feed-forward (SwiGLU)
    w1_out = x_norm @ w1_weight.T  # (batch, seq_len, d_ff)
    w3_out = x_norm @ w3_weight.T  # (batch, seq_len, d_ff)
    
    # Apply SiLU to w3
    gate = F.silu(w3_out)
    combined = gate * w1_out
    
    ffn_out = combined @ w2_weight.T  # (batch, seq_len, d_model)
    
    # Second residual
    output = x + ffn_out
    
    return output

def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """
    Similar to before it uses RoPE
    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.

        context_length (int): The maximum number of tokens to process at once.

        d_model (int): The dimensionality of the model embeddings and sublayer outputs.

        num_layers (int): The number of Transformer layers to use.

        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.

        d_ff (int): Dimensionality of the feed-forward inner layer (See Vaswani et al).

        rope_theta (float): The RoPE `theta` parameter.

        weights (dict[str, Tensor]):
            {num_layers} refers to an integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    raise NotImplementedError
