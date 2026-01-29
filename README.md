<div align="center">

# IC-Effect: Precise and Efficient Video Effects Editing via In-Context Learning

</div>

<div align="center">

<a href="https://github.com/KoTion">Yuanhang Li</a><sup>1</sup>,
<a href="https://scholar.google.com/citations?user=L2YS0jgAAAAJ&hl=en">Yiren Song</a><sup>2</sup>,
<a >Junzhe Bai</a><sup>1</sup>,
<a >Xinran Liang</a><sup>1</sup>,
<a href="https://github.com/Tigeryang93">Hu Yang</a><sup>3</sup>,
<a href="https://dblp.org/pid/148/1665.html">Libiao Jin</a><sup>1</sup>,
<a href="https://scholar.google.com/citations?user=VTQZF6EAAAAJ&hl=zh-CN">Qi Mao</a><sup>1,<i class="fas fa-envelope" style="font-size: 0.8em; vertical-align: text-top;"></i></sup>

<sup>1</sup>School of Information and Communication Engineering, Communication University of China</span>
<sup>2</sup>ShowLab, National University of Singapore</span>
<sup>3</sup>Baidu Inc.</span>

[![Project Website](https://img.shields.io/badge/Project-Website-orange
)](https://cuc-mipg.github.io/IC-Effect/)
<a href="https://arxiv.org/abs/2512.15635"><img src="https://img.shields.io/badge/arXiv-2512.15635-A42C25.svg" alt="arXiv"></a>
<a href="https://huggingface.co/CUC-MIPG/IC-Effect"><img src="https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg" alt="HuggingFace"></a>
<a href="https://huggingface.co/datasets/CUC-MIPG/IC-Effect/"><img src="https://img.shields.io/badge/🤗_HuggingFace-Dataset-ffbd45.svg" alt="HuggingFace"></a>

</div>  

<img src='./assets/teaser.png' width='100%' />




## Abstract
TL; DR: **IC-Effect** is the first instruction-guied video VFX editing framework.

<details><summary>CLICK for the full abstract</summary>
We propose IC-Effect, an instruction-guided, DiT-based framework for few-shot video VFX editing that synthesizes complex effects (e.g., flames, particles and cartoon characters) while strictly preserving spatial and temporal consistency. Video VFX editing is highly challenging because injected effects must blend seamlessly with the background, the background must remain entirely unchanged, and effect patterns must be learned efficiently from limited paired data. However, existing video editing models fail to satisfy these requirements. IC-Effect leverages the source video as clean contextual conditions, exploiting the contextual learning capability of DiT models to achieve precise background preservation and natural effect injection. A two-stage training strategy, consisting of general editing adaptation followed by effect-specific learning via EffectLoRA, ensures strong instruction following and robust effect modeling. To further improve efficiency, we introduce spatiotemporal sparse tokenization, enabling high fidelity with substantially reduced computation. We also release a paired VFX editing dataset spanning 15 high-quality visual styles. Extensive experiments show that IC-Effect delivers high-quality, controllable, and temporally consistent VFX editing, opening new possibilities for video creation.
</details>

## 💡 Changelog
- 2026.1.29 Release Inference code, VideoEditor weights and Benchmark!
- 2025.12.17 Release Project Page and Paper!


## 📑 Todo List:
- [x] Release Inference code
- [x] Release weights of Video-Editor 
- [x] Release Benchmark
- [ ] Release VFX Dataset
- [ ] Release weights of Effect-LoRA
- [ ] Release Training code


## Getting Started with IC-Effect

### 1. **Environment setup**
```bash
git clone https://github.com/CUC-MIPG/IC-Effect.git
cd IC-Effect

conda create -n IC-Effect python=3.10
conda activate IC-Effect
```
### 2. **Requirements installation**
```bash
pip install requirements.txt
```

### 3. **Download pre-trained models**

We use Wan2.2-T2V-A14B as backbone, please download from [HuggingFace](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers) or using huggingface-cli:


```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B-Diffusers --local-dir ./Wan-AI/Wan2.2-I2V-A14B-Diffusers
```

You can download the VideoEditor trained checkpoints for the IC-Effect model from [HuggingFace](https://huggingface.co/CUC-MIPG/IC-Effect) for common video editing.


### 4. **Inference**
```bash
python src/inference_2.py \
    --model_id [your_model_dir] \
    --ckpt_path [your_ckpt_dir] \
    --condition_path [you_video_path] \
    --prompts [you_edit_instruction] 
```


## Citation
```
@article{li2025iceffect,
    title={IC-Effect: Precise and Efficient Video Effects Editing via In-Context Learning},
    author={Yuanhang Li and Yiren Song and Junzhe Bai and Xinran Liang and Hu Yang and Libiao Jin and Qi Mao},
    journal={arXiv preprint arXiv:2512.15635},
    year={2025}
  }
```