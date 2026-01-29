import torch
from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from diffusers import FlowMatchEulerDiscreteScheduler
from models.transformer_wan import WanTransformer3DModel
from models.custom_pipeline import CustomWanPipeline as WanPipeline


model_id = "/path-to-model/Wan2.2-T2V-A14B-Diffusers"

transformer = WanTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.float32
        )

from peft import LoraConfig
transformer_lora_config = LoraConfig(
                r=96, lora_alpha=96, init_lora_weights=True,
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )

transformer.add_adapter(transformer_lora_config)

tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = UMT5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )

        # VAE
vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.bfloat16
        )
base_sampler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
sample_scheduler = UniPCMultistepScheduler.from_config(
            base_sampler.config, flow_shift=5
        )


def _load_lora_from_ckpt(pipe, ckpt_path, device):

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

pipeline = WanPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        scheduler=sample_scheduler,
        )

_load_lora_from_ckpt(pipeline, "/path-to-lora/", device='cpu')
pipeline.fuse_lora()
pipeline.unload_lora_weights()
pipeline.save_pretrained("/path-to-save/")


