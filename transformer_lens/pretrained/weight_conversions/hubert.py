import einops
import torch
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_hubert_weights(hf_model, cfg: HookedTransformerConfig):
    """
    Convert a Hugging Face HuBERT model's transformer encoder weights
    into a TransformerLens-compatible state_dict.

    This ignores HuBERT's convolutional feature extractor and feature projection,
    since we assume they are handled externally (e.g., via hf_model.feature_extractor
    and hf_model.feature_projection).

    Args:
        hf_model: A pretrained HuggingFace HuBERT model (e.g., HubertModel.from_pretrained(...))
        cfg: TransformerLens HookedTransformerConfig

    Returns:
        state_dict: a dict mapping TransformerLens parameter names to torch tensors
                    suitable for model.load_state_dict(state_dict, strict=False)
    """
    state_dict = {}

    # Shortcut to encoder layers
    encoder_layers = hf_model.encoder.layers

    for l, layer in enumerate(encoder_layers):
        # --- Self-attention projections ---
        q_proj = layer.self_attn.q_proj.weight
        k_proj = layer.self_attn.k_proj.weight
        v_proj = layer.self_attn.v_proj.weight
        out_proj = layer.self_attn.out_proj.weight

        # Reshape Q, K, V into [n_heads, d_model, d_head]
        d_model = cfg.d_model
        n_heads = cfg.n_heads
        d_head = d_model // n_heads

        state_dict[f"blocks.{l}.attn.W_Q"] = einops.rearrange(
            q_proj, "(n h) m -> n m h", n=n_heads
        )
        state_dict[f"blocks.{l}.attn.W_K"] = einops.rearrange(
            k_proj, "(n h) m -> n m h", n=n_heads
        )
        state_dict[f"blocks.{l}.attn.W_V"] = einops.rearrange(
            v_proj, "(n h) m -> n m h", n=n_heads
        )
        state_dict[f"blocks.{l}.attn.W_O"] = einops.rearrange(
            out_proj, "m (n h) -> n h m", n=n_heads
        )

        # --- LayerNorms ---
        state_dict[f"blocks.{l}.ln1.w"] = layer.layer_norm.weight
        state_dict[f"blocks.{l}.ln1.b"] = layer.layer_norm.bias
        state_dict[f"blocks.{l}.ln2.w"] = layer.final_layer_norm.weight
        state_dict[f"blocks.{l}.ln2.b"] = layer.final_layer_norm.bias

        # --- Feed-forward (MLP) ---
        fc1 = layer.fc1.weight
        fc2 = layer.fc2.weight
        fc1_bias = layer.fc1.bias
        fc2_bias = layer.fc2.bias

        state_dict[f"blocks.{l}.mlp.W_in"] = fc1.T  # shape [d_model, d_mlp]
        state_dict[f"blocks.{l}.mlp.b_in"] = fc1_bias
        state_dict[f"blocks.{l}.mlp.W_out"] = fc2.T  # shape [d_mlp, d_model]
        state_dict[f"blocks.{l}.mlp.b_out"] = fc2_bias

    return state_dict
