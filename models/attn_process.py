import torch
from typing import Optional
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention


class ConditionAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        rotary_emb2: Optional[torch.Tensor] = None,
        rotary_emb3: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # [B, L, D] -> [B, H, L, Dh]
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def _canonicalize_cos_sin(freqs_cos: torch.Tensor, freqs_sin: torch.Tensor, L: int, C: int, ref: torch.dtype, dev: torch.device):

                cos = freqs_cos.to(device=dev, dtype=ref)
                sin = freqs_sin.to(device=dev, dtype=ref)

                while cos.dim() > 2:
                    cos = cos.select(0, 0)
                    sin = sin.select(0, 0)

                assert cos.dim() == 2 and sin.dim() == 2, f"Unexpected cos/sin dim: {cos.shape} / {sin.shape}"

                if cos.size(-1) == 2 * C:
                    cos = cos[..., 0::2]
                    sin = sin[..., 1::2]
                elif cos.size(-1) != C:
                    
                    if cos.size(-1) > C:
                        cos = cos[..., :C]
                        sin = sin[..., :C]
                    else:
                        pad = C - cos.size(-1)
                        cos = torch.cat([cos, cos[..., -1:].expand(cos.size(0), pad)], dim=-1)
                        sin = torch.cat([sin, sin[..., -1:].expand(sin.size(0), pad)], dim=-1)

                if cos.size(-2) > L:
                    cos = cos[:L, :]
                    sin = sin[:L, :]
                elif cos.size(-2) < L:
                    pad = L - cos.size(-2)
                    cos = torch.cat([cos, cos[-1:, :].expand(pad, cos.size(1))], dim=-2)
                    sin = torch.cat([sin, sin[-1:, :].expand(pad, sin.size(1))], dim=-2)

                cos = cos.view(1, 1, L, C).contiguous()
                sin = sin.view(1, 1, L, C).contiguous()
                return cos, sin

            def apply_rotary_emb(h: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
                # h: [B, H, L, Dh]
                B, Hh, L, Dh = h.shape
                C = Dh // 2

                x = h.view(B, Hh, L, C, 2)
                x1, x2 = x[..., 0], x[..., 1]  # [B, H, L, C]

                cos, sin = _canonicalize_cos_sin(freqs_cos, freqs_sin, L=L, C=C, ref=h.dtype, dev=h.device)

                out = torch.empty_like(h)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out

            if rotary_emb2 is not None:
                if rotary_emb3 is not None:
                    L1 = rotary_emb[0].shape[-2]
                    L2 = rotary_emb2[0].shape[-2]
                    L3 = rotary_emb3[0].shape[-2]
                    query[:, :, :L1] = apply_rotary_emb(query[:, :, :L1], *rotary_emb)
                    key[:, :, :L1] = apply_rotary_emb(key[:, :, :L1], *rotary_emb)
                    query[:, :, L1:L1 + L2] = apply_rotary_emb(query[:, :, L1:L1 + L2], *rotary_emb2)
                    key[:, :, L1:L1 + L2] = apply_rotary_emb(key[:, :, L1:L1 + L2], *rotary_emb2)
                    query[:, :, -L3:] = apply_rotary_emb(query[:, :, -L3:], *rotary_emb3)
                    key[:, :, -L3:] = apply_rotary_emb(key[:, :, -L3:], *rotary_emb3)
                else:
                    half = query.shape[2] // 2
                    query[:, :, :half] = apply_rotary_emb(query[:, :, :half], *rotary_emb)
                    key[:, :, :half] = apply_rotary_emb(key[:, :, :half], *rotary_emb)
                    query[:, :, half:] = apply_rotary_emb(query[:, :, half:], *rotary_emb2)
                    key[:, :, half:] = apply_rotary_emb(key[:, :, half:], *rotary_emb2)
            else:
                query = apply_rotary_emb(query, *rotary_emb)
                key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
