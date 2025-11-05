"""Hooked Encoder.

Contains a BERT style model. This is separate from :class:`transformer_lens.HookedTransformer`
because it has a significantly different architecture to e.g. GPT style transformers.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, overload

import torch
import torch.nn as nn
from einops import repeat
from jaxtyping import Float, Int
from transformers.models.auto.tokenization_auto import AutoTokenizer
from typing_extensions import Literal

import transformer_lens.loading_from_pretrained as loading
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.components import (
    MLP,
    Attention,
    BertBlock,
    BertEmbed,
    BertMLMHead,
    BertNSPHead,
    BertPooler,
    Unembed,
)
from transformer_lens.FactoredMatrix import FactoredMatrix
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utilities import devices

T = TypeVar("T", bound="HookedEncoder")


class HookedEncoder(HookedRootModule):
    """
    This class implements a BERT-style encoder using the components in ./components.py, with HookPoints on every interesting activation. It inherits from HookedRootModule.

    Limitations:
    - The model does not include dropouts, which may lead to inconsistent results from training or fine-tuning.

    Like HookedTransformer, it can have a pretrained Transformer's weights loaded via `.from_pretrained`. There are a few features you might know from HookedTransformer which are not yet supported:
        - There is no preprocessing (e.g. LayerNorm folding) when loading a pretrained model
    """

    def __init__(
        self,
        cfg: Union[HookedTransformerConfig, Dict],
        tokenizer: Optional[Any] = None,
        move_to_device: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig(**cfg)
        elif isinstance(cfg, str):
            raise ValueError(
                "Please pass in a config dictionary or HookedTransformerConfig object. If you want to load a pretrained model, use HookedEncoder.from_pretrained() instead."
            )
        self.cfg = cfg

        assert self.cfg.n_devices == 1, "Multiple devices not supported for HookedEncoder"
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif self.cfg.tokenizer_name is not None:
            huggingface_token = os.environ.get("HF_TOKEN", "")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.tokenizer_name,
                token=huggingface_token if len(huggingface_token) > 0 else None,
            )
        else:
            self.tokenizer = None

        if self.cfg.d_vocab == -1:
            # If we have a tokenizer, vocab size can be inferred from it.
            assert self.tokenizer is not None, "Must provide a tokenizer if d_vocab is not provided"
            self.cfg.d_vocab = max(self.tokenizer.vocab.values()) + 1
        if self.cfg.d_vocab_out == -1:
            self.cfg.d_vocab_out = self.cfg.d_vocab

        self.embed = BertEmbed(self.cfg)
        self.blocks = nn.ModuleList([BertBlock(self.cfg) for _ in range(self.cfg.n_layers)])
        self.mlm_head = BertMLMHead(self.cfg)
        self.unembed = Unembed(self.cfg)
        self.nsp_head = BertNSPHead(self.cfg)
        self.pooler = BertPooler(self.cfg)

        self.hook_full_embed = HookPoint()

        if move_to_device:
            if self.cfg.device is None:
                raise ValueError("Cannot move to device when device is None")
            self.to(self.cfg.device)

        self.setup()

    def encoder_output(
        self,
        frames: torch.Tensor,           # (batch, frames, d_model)   <-- precomputed conv features
        one_zero_attention_mask: Optional[torch.Tensor] = None,  # (batch, frames)
    ):
        # Ensure device
        if frames.device.type != self.cfg.device:
            frames = frames.to(self.cfg.device)
            if one_zero_attention_mask is not None:
                one_zero_attention_mask = one_zero_attention_mask.to(self.cfg.device)
    
        # directly use frames as "embed output" (skip to_tokens/embed)
        resid = self.hook_full_embed(frames)
    
        large_negative_number = -torch.inf
        mask = (
            repeat(1 - one_zero_attention_mask, "batch pos -> batch 1 1 pos")
            if one_zero_attention_mask is not None
            else None
        )
        additive_attention_mask = (
            torch.where(mask == 1, large_negative_number, 0) if mask is not None else None
        )
    
        for block in self.blocks:
            resid = block(resid, additive_attention_mask)
    
        return resid

    def forward(
        self,
        input,  # either: Tensor[batch, samples] (raw wave) OR Tensor[batch, frames, feat_dim] (precomputed conv features)
        return_type: Optional[str] = "logits",  # "logits" or None or "hidden"
        lengths: Optional[torch.Tensor] = None,  # optional lengths in frames (for padding), shape [batch]
        masked_positions: Optional[torch.BoolTensor] = None,  # optional mask of positions to replace with masked_spec_embed [batch, frames]
        preprocess_already: bool = False,  # if True, input is precomputed frames
    ):
        """
        HuBERT-like forward. If preprocess_already=False, expects raw audio waveforms (batch, samples)
        and runs feature_extractor -> feature_projection. If preprocess_already=True, expects
        (batch, frames, feat_dim) already projected to model hidden dim (or if feat_dim != hidden, we project).
        """
    
        device = self.cfg.device
    
        # 1) Build feature frames
        if preprocess_already:
            # assume input is frames, possibly already in d_model
            features = input.to(device)
            # if feature dim != model hidden, optionally project (defensive)
            if features.shape[-1] != self.cfg.d_model:
                raise ValueError(f"features shape is incorrect. Model is expecting {self.cfg.d_model}, but get {features.shape[-1]}")
        else:
            # raw waveform path
            # feature_extractor returns something like (batch, feat_len, feat_dim) or (batch, feat_len)
            # Hugging Face: feature_extractor expects float waveform batched
            wave = input.to(device)
            features = self.feature_extractor(wave)               # conv layers -> torch.float
            features = self.feature_projection(features)         # linear + layernorm -> (batch, frames, d_model)
    
        # 2) Optionally apply masked_spec_embed for masked_positions
        # masked_positions: bool tensor [batch, frames] where True indicates masked frames
        if masked_positions is not None:
            # masked_spec_embed is shape (d_model,)
            mask = masked_positions.to(device)
            masked_vec = self.masked_spec_embed.view(1, 1, -1)  # (1,1,d_model)
            features = torch.where(mask.unsqueeze(-1), masked_vec, features)
    
        # 3) Build attention mask from lengths if provided; else assume all ones
        if lengths is not None:
            # lengths in frames; create one_zero_attention_mask with 1 for valid / 0 for padding
            max_frames = features.shape[1]
            rng = torch.arange(max_frames, device=device).unsqueeze(0)  # (1, frames)
            one_zero_attention_mask = (rng < lengths.unsqueeze(1)).long()  # (batch, frames)
        else:
            one_zero_attention_mask = torch.ones(features.shape[:2], dtype=torch.long, device=device)
    
        # 4) Pass through (possibly identical) encoder routine
        # For the HookedTransformer code you had: resid = self.hook_full_embed(self.embed(tokens, ...))
        # For HuBERT we treat 'features' as the residual input.
        resid = self.encoder_output(features, one_zero_attention_mask)
      
        # 5) Prediction head: project hidden states to logits/predictions over discrete units
        if return_type == "hidden":
            return resid   # (batch, frames, d_model)
    
        # project_hid -> predictions (frame-wise)
        pred = self.project_hid(resid)  # shape (batch, frames, target_dim) or (batch, frames, n_classes)
        # If your project_hid produces vectors and you want logits over cluster ids, there may be an extra linear/unembed
        # e.g., logits = self.unembed(pred) or pred itself already logits.
    
        if return_type == "logits" or return_type is None:
            return pred
    
        return None


    @overload
    def run_with_cache(
        self, *model_args: Any, return_cache_object: Literal[True] = True, **kwargs: Any
    ) -> Tuple[Float[torch.Tensor, "batch pos d_vocab"], ActivationCache]:
        ...

    @overload
    def run_with_cache(
        self, *model_args: Any, return_cache_object: Literal[False], **kwargs: Any
    ) -> Tuple[Float[torch.Tensor, "batch pos d_vocab"], Dict[str, torch.Tensor]]:
        ...

    def run_with_cache(
        self,
        *model_args: Any,
        return_cache_object: bool = True,
        remove_batch_dim: bool = False,
        **kwargs: Any,
    ) -> Tuple[
        Float[torch.Tensor, "batch pos d_vocab"],
        Union[ActivationCache, Dict[str, torch.Tensor]],
    ]:
        """
        Wrapper around run_with_cache in HookedRootModule. If return_cache_object is True, this will return an ActivationCache object, with a bunch of useful HookedTransformer specific methods, otherwise it will return a dictionary of activations as in HookedRootModule. This function was copied directly from HookedTransformer.
        """
        out, cache_dict = super().run_with_cache(
            *model_args, remove_batch_dim=remove_batch_dim, **kwargs
        )
        if return_cache_object:
            cache = ActivationCache(cache_dict, self, has_batch_dim=not remove_batch_dim)
            return out, cache
        else:
            return out, cache_dict

    def to(  # type: ignore
        self,
        device_or_dtype: Union[torch.device, str, torch.dtype],
        print_details: bool = True,
    ):
        return devices.move_to_and_update_config(self, device_or_dtype, print_details)

    def cuda(self: T, device: Optional[Union[int, torch.device]] = None) -> T:
        if isinstance(device, int):
            return self.to(f"cuda:{device}")
        elif device is None:
            return self.to("cuda")
        else:
            return self.to(device)

    def cpu(self: T) -> T:
        return self.to("cpu")

    def mps(self: T) -> T:
        return self.to(torch.device("mps"))

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        checkpoint_index: Optional[int] = None,
        checkpoint_value: Optional[int] = None,
        hf_model: Optional[Any] = None,
        device: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        move_to_device: bool = True,
        dtype: torch.dtype = torch.float32,
        **from_pretrained_kwargs: Any,
    ) -> HookedEncoder:
        """Loads in the pretrained weights from huggingface. Currently supports loading weight from HuggingFace BertForMaskedLM. Unlike HookedTransformer, this does not yet do any preprocessing on the model."""
        logging.warning(
            "Support for BERT in TransformerLens is currently experimental, until such a time when it has feature "
            "parity with HookedTransformer and has been tested on real research tasks. Until then, backward "
            "compatibility is not guaranteed. Please see the docs for information on the limitations of the current "
            "implementation."
            "\n"
            "If using BERT for interpretability research, keep in mind that BERT has some significant architectural "
            "differences to GPT. For example, LayerNorms are applied *after* the attention and MLP components, meaning "
            "that the last LayerNorm in a block cannot be folded."
        )

        assert not (
            from_pretrained_kwargs.get("load_in_8bit", False)
            or from_pretrained_kwargs.get("load_in_4bit", False)
        ), "Quantization not supported"

        if "torch_dtype" in from_pretrained_kwargs:
            dtype = from_pretrained_kwargs["torch_dtype"]

        official_model_name = loading.get_official_model_name(model_name)

        cfg = loading.get_pretrained_model_config(
            official_model_name,
            checkpoint_index=checkpoint_index,
            checkpoint_value=checkpoint_value,
            fold_ln=False,
            device=device,
            n_devices=1,
            dtype=dtype,
            **from_pretrained_kwargs,
        )

        state_dict = loading.get_pretrained_state_dict(
            official_model_name, cfg, hf_model, dtype=dtype, **from_pretrained_kwargs
        )

        model = cls(cfg, tokenizer, move_to_device=False)

        model.load_state_dict(state_dict, strict=False)

        if move_to_device:
            model.to(cfg.device)

        print(f"Loaded pretrained model {model_name} into HookedEncoder")

        return model

    @property
    def W_U(self) -> Float[torch.Tensor, "d_model d_vocab"]:
        """
        Convenience to get the unembedding matrix (ie the linear map from the final residual stream to the output logits)
        """
        return self.unembed.W_U

    @property
    def b_U(self) -> Float[torch.Tensor, "d_vocab"]:
        """
        Convenience to get the unembedding bias
        """
        return self.unembed.b_U

    @property
    def W_E(self) -> Float[torch.Tensor, "d_vocab d_model"]:
        """
        Convenience to get the embedding matrix
        """
        return self.embed.embed.W_E

    @property
    def W_pos(self) -> Float[torch.Tensor, "n_ctx d_model"]:
        """
        Convenience function to get the positional embedding. Only works on models with absolute positional embeddings!
        """
        return self.embed.pos_embed.W_pos

    @property
    def W_E_pos(self) -> Float[torch.Tensor, "d_vocab+n_ctx d_model"]:
        """
        Concatenated W_E and W_pos. Used as a full (overcomplete) basis of the input space, useful for full QK and full OV circuits.
        """
        return torch.cat([self.W_E, self.W_pos], dim=0)

    @property
    def W_K(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stacks the key weights across all layers"""
        for block in self.blocks:
            assert isinstance(block.attn, Attention)
        return torch.stack([block.attn.W_K for block in self.blocks], dim=0)

    @property
    def W_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stacks the query weights across all layers"""
        for block in self.blocks:
            assert isinstance(block.attn, Attention)
        return torch.stack([block.attn.W_Q for block in self.blocks], dim=0)

    @property
    def W_V(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stacks the value weights across all layers"""
        for block in self.blocks:
            assert isinstance(block.attn, Attention)
        return torch.stack([block.attn.W_V for block in self.blocks], dim=0)

    @property
    def W_O(self) -> Float[torch.Tensor, "n_layers n_heads d_head d_model"]:
        """Stacks the attn output weights across all layers"""
        for block in self.blocks:
            assert isinstance(block.attn, Attention)
        return torch.stack([block.attn.W_O for block in self.blocks], dim=0)

    @property
    def W_in(self) -> Float[torch.Tensor, "n_layers d_model d_mlp"]:
        """Stacks the MLP input weights across all layers"""
        for block in self.blocks:
            assert isinstance(block.mlp, MLP)
        return torch.stack([block.mlp.W_in for block in self.blocks], dim=0)

    @property
    def W_out(self) -> Float[torch.Tensor, "n_layers d_mlp d_model"]:
        """Stacks the MLP output weights across all layers"""
        for block in self.blocks:
            assert isinstance(block.mlp, MLP)
        return torch.stack([block.mlp.W_out for block in self.blocks], dim=0)

    @property
    def b_K(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stacks the key biases across all layers"""
        for block in self.blocks:
            assert isinstance(block.attn, Attention)
        return torch.stack([block.attn.b_K for block in self.blocks], dim=0)

    @property
    def b_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stacks the query biases across all layers"""
        for block in self.blocks:
            assert isinstance(block.attn, Attention)
        return torch.stack([block.attn.b_Q for block in self.blocks], dim=0)

    @property
    def b_V(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stacks the value biases across all layers"""
        for block in self.blocks:
            assert isinstance(block.attn, Attention)
        return torch.stack([block.attn.b_V for block in self.blocks], dim=0)

    @property
    def b_O(self) -> Float[torch.Tensor, "n_layers d_model"]:
        """Stacks the attn output biases across all layers"""
        for block in self.blocks:
            assert isinstance(block.attn, Attention)
        return torch.stack([block.attn.b_O for block in self.blocks], dim=0)

    @property
    def b_in(self) -> Float[torch.Tensor, "n_layers d_mlp"]:
        """Stacks the MLP input biases across all layers"""
        for block in self.blocks:
            assert isinstance(block.mlp, MLP)
        return torch.stack([block.mlp.b_in for block in self.blocks], dim=0)

    @property
    def b_out(self) -> Float[torch.Tensor, "n_layers d_model"]:
        """Stacks the MLP output biases across all layers"""
        for block in self.blocks:
            assert isinstance(block.mlp, MLP)
        return torch.stack([block.mlp.b_out for block in self.blocks], dim=0)

    @property
    def QK(self) -> FactoredMatrix:  # [n_layers, n_heads, d_model, d_model]
        """Returns a FactoredMatrix object with the product of the Q and K matrices for each layer and head.
        Useful for visualizing attention patterns."""
        return FactoredMatrix(self.W_Q, self.W_K.transpose(-2, -1))

    @property
    def OV(self) -> FactoredMatrix:  # [n_layers, n_heads, d_model, d_model]
        """Returns a FactoredMatrix object with the product of the O and V matrices for each layer and head."""
        return FactoredMatrix(self.W_V, self.W_O)

    def all_head_labels(self) -> List[str]:
        """Returns a list of strings with the format "L{l}H{h}", where l is the layer index and h is the head index."""
        return [f"L{l}H{h}" for l in range(self.cfg.n_layers) for h in range(self.cfg.n_heads)]
