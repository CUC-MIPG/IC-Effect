from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.normalization import FP32LayerNorm


class WanImageEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, pos_embed_seq_len=None):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)
        if pos_embed_seq_len is not None:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_seq_len, in_features))
        else:
            self.pos_embed = None

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            batch_size, seq_len, embed_dim = encoder_hidden_states_image.shape
            encoder_hidden_states_image = encoder_hidden_states_image.view(-1, 2 * seq_len, embed_dim)
            encoder_hidden_states_image = encoder_hidden_states_image + self.pos_embed

        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
        pos_embed_seq_len: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        timestep_seq_len: Optional[int] = None,
        ctrl_embeds: Optional[torch.Tensor] = None,
    ):
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)

        if ctrl_embeds is not None:
            timestep_proj = self.time_proj(self.act_fn(temb) + self.act_fn(ctrl_embeds))
        else:
            timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim
        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64

        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def forward(self, hidden_states: torch.Tensor, scale_factor: Optional[Tuple[int, int, int]] = None) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        split_sizes = [
            self.attention_head_dim - 2 * (self.attention_head_dim // 3),
            self.attention_head_dim // 3,
            self.attention_head_dim // 3,
        ]

        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        freqs_cos_f = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        if scale_factor is not None:
            freqs_cos_f_c = torch.cat([freqs_cos[0][i].unsqueeze(0) for i in range(0, ppf, scale_factor[0])]).view(
                ppf // scale_factor[0], 1, 1, -1).expand(ppf // scale_factor[0], pph // scale_factor[1], ppw // scale_factor[2], -1)
            freqs_cos_h_c = torch.cat([freqs_cos[1][i].unsqueeze(0) for i in range(0, pph, scale_factor[1])]).view(
                1, pph // scale_factor[1], 1, -1).expand(ppf // scale_factor[0], pph // scale_factor[1], ppw // scale_factor[2], -1)
            freqs_cos_w_c = torch.cat([freqs_cos[2][i].unsqueeze(0) for i in range(0, ppw, scale_factor[2])]).view(
                1, 1, ppw // scale_factor[2], -1).expand(ppf // scale_factor[0], pph // scale_factor[1], ppw // scale_factor[2], -1)

            freqs_sin_f_c = torch.cat([freqs_sin[0][i].unsqueeze(0) for i in range(0, ppf, scale_factor[0])]).view(
                ppf // scale_factor[0], 1, 1, -1).expand(ppf // scale_factor[0], pph // scale_factor[1], ppw // scale_factor[2], -1)
            freqs_sin_h_c = torch.cat([freqs_sin[1][i].unsqueeze(0) for i in range(0, pph, scale_factor[1])]).view(
                1, pph // scale_factor[1], 1, -1).expand(ppf // scale_factor[0], pph // scale_factor[1], ppw // scale_factor[2], -1)
            freqs_sin_w_c = torch.cat([freqs_sin[2][i].unsqueeze(0) for i in range(0, ppw, scale_factor[2])]).view(
                1, 1, ppw // scale_factor[2], -1).expand(ppf // scale_factor[0], pph // scale_factor[1], ppw // scale_factor[2], -1)

            freqs_cos_c = torch.cat([freqs_cos_f_c, freqs_cos_h_c, freqs_cos_w_c], dim=-1).reshape(1, 1, ppf // scale_factor[0] * pph // scale_factor[1] * ppw // scale_factor[2], -1)
            freqs_sin_c = torch.cat([freqs_sin_f_c, freqs_sin_h_c, freqs_sin_w_c], dim=-1).reshape(1, 1, ppf // scale_factor[0] * pph // scale_factor[1] * ppw // scale_factor[2], -1)

            return freqs_cos_c, freqs_sin_c


        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)

        return freqs_cos, freqs_sin
