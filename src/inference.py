import os
import argparse
import copy
import warnings
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from decord import VideoReader
from torchvision import transforms
from PIL import Image

import pytorch_lightning as L
from pytorch_lightning.utilities import rank_zero_only

from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from diffusers.utils import export_to_video
from diffusers import FlowMatchEulerDiscreteScheduler

from models.transformer_wan import WanTransformer3DModel
from models.custom_pipeline import CustomWanPipeline as WanPipeline
from models.attn_process import ConditionAttnProcessor2_0


@rank_zero_only
def silence_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ICEffect_Infer(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.hparams = opt
        self.is_configured = False

    def configure_model(self):
        if self.is_configured:
            return
        self.is_configured = True

        model_id = self.hparams.model_id

        # tokenizer / text encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )

        # VAE
        self.vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.bfloat16
        )

        base_sampler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        self.sample_scheduler = UniPCMultistepScheduler.from_config(
            base_sampler.config, flow_shift=5
        )

        # transformer
        self.transformer = WanTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16
        )

        # 冻结主干
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.transformer.requires_grad_(False)

        self.transformer.gradient_checkpointing = True
        self.transformer.enable_gradient_checkpointing()

        # latents 标准化参数
        self.register_buffer(
            'latents_mean',
            torch.tensor(self.vae.config.latents_mean).float().view(1, self.vae.config.z_dim, 1, 1, 1),
            persistent=False
        )
        self.register_buffer(
            'latents_std',
            torch.tensor(self.vae.config.latents_std).float().view(1, self.vae.config.z_dim, 1, 1, 1),
            persistent=False
        )

        # 保存下 config（可用于 debug）
        self.vae_config = self.vae.config
        self.model_config = self.transformer.module.config if hasattr(self.transformer, "module") else self.transformer.config

        # LoRA（仅在 use_lora=True 时准备 adapter 容器，具体权重后续再加载）
        self.using_lora = bool(self.hparams.use_lora)
        if self.using_lora:
            from peft import LoraConfig
            transformer_lora_config = LoraConfig(
                r=self.hparams.lora_rank, lora_alpha=self.hparams.lora_rank, init_lora_weights=True,
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            self.transformer.add_adapter(transformer_lora_config)

        # 设置 ConditionAttnProcessor，与训练端一致
        for blk in self.transformer.blocks:
            blk.attn1.set_processor(ConditionAttnProcessor2_0())

        # 额外 patch embedding（与训练端一致）
        self.transformer.patch_embedding_extra = copy.deepcopy(self.transformer.patch_embedding).requires_grad_(True)

    def _load_lora_from_ckpt(self, ckpt_path, device):

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd_all = ckpt["state_dict"]

        # LoRA/attn processor
        if "transformer_processor" in sd_all:
            sd = sd_all["transformer_processor"]
            cur = self.transformer.state_dict()
            filtered = {k: v for k, v in sd.items() if (k in cur and cur[k].shape == v.shape)}
            skipped = [k for k in sd.keys() if k not in filtered]
            print(f"[Infer][LoRA] Load {len(filtered)}/{len(sd)} keys. Skipped {len(skipped)} mismatched keys.")
            missing_after = set(cur.keys()) - set(filtered.keys())
            self.transformer.load_state_dict(filtered, strict=False)
        else:
            print("[Infer] 'transformer_processor' not found in ckpt.state_dict; skip LoRA.")

        # patch_embedding_extra
        if "patch_embedding_extra" in sd_all:
            sd2 = sd_all["patch_embedding_extra"]
            cur2 = self.transformer.state_dict()
            filtered2 = {k: v for k, v in sd2.items() if (k in cur2 and cur2[k].shape == v.shape)}
            skipped2 = [k for k in sd2.keys() if k not in filtered2]
            print(f"[Infer][patch_embedding_extra] Load {len(filtered2)}/{len(sd2)} keys. Skipped {len(skipped2)} mismatched keys.")
            self.transformer.load_state_dict(filtered2, strict=False)
        else:
            print("[Infer] 'patch_embedding_extra' not found in ckpt.state_dict; skip.")

    def get_frames_num(self, condition_path):
        train_video_transforms = transforms.Compose(
            [
                transforms.Resize((self.hparams.height, self.hparams.width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        train_video_transforms_down = transforms.Compose(
            [
                transforms.Resize((int(self.hparams.height/2), int(self.hparams.width/2))),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        

        video_reader = VideoReader(condition_path)
        video_length = len(video_reader)
        stride = 1
        start_index = 0
        if video_length < 81:
            frame_indices = start_index + np.arange(video_length) * stride
        else:
            video_length = 81
            frame_indices = start_index + np.arange(video_length) * stride
        
        video = video_reader.get_batch(frame_indices).asnumpy()  # F, H, W, C
        video = [Image.fromarray(frame) for frame in video]
        first_frame = video[0]
        pixel_values = [train_video_transforms_down(frame) for frame in video]
        pixel_values = torch.stack(pixel_values)  # F, C, H, W
        first_frame = train_video_transforms(first_frame)

        return pixel_values.permute(1, 0, 2, 3), first_frame, video_length
        
        

    @torch.no_grad()
    def run_infer(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.configure_model()
        self.to(device)

        if self.using_lora:
            ckpt_path = getattr(self.hparams, "ckpt_path", None)
            self._load_lora_from_ckpt(ckpt_path, device)

        # 采样管线
        pipeline = WanPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            scheduler=self.sample_scheduler,
        )
        save_root = os.path.join(self.hparams.output_root)
        os.makedirs(save_root, exist_ok=True)
        print(f"[Infer] Save to: {save_root}")

        # 准备数据
        condition_video, first_frame, video_length = self.get_frames_num(self.hparams.condition_path)
        prompts = self.hparams.prompts

        first_frames = first_frame.to(device).unsqueeze(0).unsqueeze(2)  # [1, C, 1, H, W]
        
        condition_video = condition_video.unsqueeze(0).to(device)


        condition_video_lat = self.vae.encode(condition_video.to(dtype=torch.bfloat16)).latent_dist.sample()
        condition_video_lat = (condition_video_lat - self.latents_mean) / self.latents_std

        first_frames_lat = self.vae.encode(first_frames.to(dtype=torch.bfloat16)).latent_dist.sample()
        first_frames_lat = (first_frames_lat - self.latents_mean) / self.latents_std

        attention_kwargs = {
            'encoder_contion_states': condition_video_lat,
            'encoder_first_states': first_frames_lat,
        }
            # 生成
        out = pipeline(
                prompt=prompts,
                height=self.hparams.height,
                width=self.hparams.width,
                num_frames=video_length,
                guidance_scale=5.0,
                attention_kwargs=attention_kwargs,
            )
        video_generate = out.frames[0]


        save_path = os.path.join(save_root, f"{self.hparams.save_name}_seed{self.hparams.seed}.mp4")
        export_to_video(video_generate, output_video_path=save_path, fps=16)
        print(f"[Infer] Saved: {save_path}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Wan2.2-T2V-A14B-Diffusers", help="model id or path of local checkpoints")
    parser.add_argument("--height", type=int, default=480, help="video height")
    parser.add_argument("--width", type=int, default=832, help="video width")
    parser.add_argument("--use_lora", type=bool, default=True, help="whether using LoRA weights")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lora_rank", type=int, default=96)
    parser.add_argument("--ckpt_path", type=str, default="", help="path to a .ckpt with LoRA/extra weights")
    parser.add_argument("--condition_path", default="", type=str, help="path to the conditioning video")
    parser.add_argument("--prompts", default="", type=str, help="text prompt")
    parser.add_argument("--output_root", default="./results/", type=str, help="where to store generated videos")
    parser.add_argument("--save_name", default="test.mp4", type=str, help="save name")

    args = parser.parse_args()
    print(args)


    L.seed_everything(args.seed)

    system = ICEffect_Infer(args)
    system.run_infer()
