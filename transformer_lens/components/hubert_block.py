class HubertBlock(nn.Module):
    """
    HuBERT-style Transformer Block (Pre-LayerNorm).
    Structurally similar to BERTBlock, but with LayerNorm applied before each sublayer.
    """

    def __init__(self, cfg: HookedTransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.attn = Attention(cfg)
        self.ln1 = LayerNorm(cfg)
        self.mlp = MLPFactory.create_mlp(self.cfg)
        self.ln2 = LayerNorm(cfg)

        self.hook_q_input = HookPoint()
        self.hook_k_input = HookPoint()
        self.hook_v_input = HookPoint()
        self.hook_attn_out = HookPoint()
        self.hook_mlp_in = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()
        self.hook_normalized_resid_post = HookPoint()

    def forward(
        self,
        resid_pre: Float[torch.Tensor, "batch pos d_model"],
        additive_attention_mask: Optional[Float[torch.Tensor, "batch 1 1 pos"]] = None,
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        resid_pre = self.hook_resid_pre(resid_pre)

        # --- Attention sublayer ---
        normed = self.ln1(resid_pre)
        attn_out = self.hook_attn_out(
            self.attn(
                self.hook_q_input(repeat_along_head_dimension(normed, self.cfg.n_heads)),
                self.hook_k_input(repeat_along_head_dimension(normed, self.cfg.n_heads)),
                self.hook_v_input(repeat_along_head_dimension(normed, self.cfg.n_heads)),
                additive_attention_mask=additive_attention_mask,
            )
        )
        resid_mid = self.hook_resid_mid(resid_pre + attn_out)

        # --- Feedforward sublayer ---
        normed_mid = self.hook_normalized_resid_post(self.ln2(resid_mid)）
        mlp_in = self.hook_mlp_in(normed_mid.clone()) if self.cfg.use_hook_mlp_in else normed_mid
        mlp_out = self.hook_mlp_out(self.mlp(mlp_in))
        resid_post = self.hook_resid_post(resid_mid + mlp_out)

        return resid_post
