# Phantom: Subject-Consistent Video Generation via Cross-Modal Alignment

<div align="center">
  
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2502.11079-b31b1b.svg)](https://arxiv.org/abs/2502.11079)&nbsp;
[![project page](https://img.shields.io/badge/Project_page-More_visualizations-green)](https://phantom-video.github.io/Phantom/)&nbsp;
<a href="https://huggingface.co/bytedance-research/Phantom"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Model&color=orange"></a>
</div>


> [**Phantom: Subject-Consistent Video Generation via Cross-Modal Alignment**](https://arxiv.org/abs/2502.11079)<br>
> [Lijie Liu](https://liulj13.github.io/)<sup> * </sup>, [Tianxiang Ma](https://tianxiangma.github.io/)<sup> * </sup>, [Bingchuan Li](https://scholar.google.com/citations?user=ac5Se6QAAAAJ)<sup> * &dagger;</sup>, [Zhuowei Chen](https://scholar.google.com/citations?user=ow1jGJkAAAAJ)<sup> * </sup>, [Jiawei Liu](https://scholar.google.com/citations?user=X21Fz-EAAAAJ), Gen Li, Siyu Zhou, [Qian He](https://scholar.google.com/citations?user=9rWWCgUAAAAJ), Xinglong Wu
> <br><sup> * </sup>Equal contribution,<sup> &dagger; </sup>Project lead
> <br>Intelligent Creation Team, ByteDance<br>

<p align="center">
<img src="assets/teaser.png" width=95%>
<p>

## 🔥 Latest News!
* Apr 10, 2025: We have updated the full version of the Phantom paper, which now includes more detailed descriptions of the model architecture and dataset pipeline.
* Apr 21, 2025: 👋 Phantom-Wan is coming! We adapted the Phantom framework into the [Wan2.1](https://github.com/Wan-Video/Wan2.1) video generation model. The inference codes and checkpoint have been released.
* Apr 23, 2025: 😊 Thanks to [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper/tree/dev) for adapting ComfyUI to Phantom-Wan-1.3B. Everyone is welcome to use it !

## 📑 Todo List
- [x] Inference codes and Checkpoint of Phantom-Wan-1.3B 
- [ ] Checkpoint of Phantom-Wan-14B
- [ ] Open source Phantom-Data
- [ ] Training codes of Phantom-Wan

## 📖 Overview
Phantom is a unified video generation framework for single and multi-subject references, built on existing text-to-video and image-to-video architectures. It achieves cross-modal alignment using text-image-video triplet data by redesigning the joint text-image injection model. Additionally, it emphasizes subject consistency in human generation while enhancing ID-preserving video generation.

## ⚡️ Quickstart

### Installation
Clone the repo:
```sh
git clone https://github.com/Phantom-video/Phantom.git
cd Phantom
```

Install dependencies:
```sh
# Ensure torch >= 2.4.0
pip install -r requirements.txt
```

### Model Download
First you need to download the 1.3B original model of Wan2.1. Download Wan2.1-1.3B using huggingface-cli:
``` sh
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B
```
Then download the Phantom-Wan-1.3B model:
``` sh
huggingface-cli download bytedance-research/Phantom --local-dir ./Phantom-Wan-1.3B
```

### Run Subject-to-Video Generation

- Single-GPU inference

``` sh
python generate.py --task s2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --phantom_ckpt ./Phantom-Wan-1.3B/Phantom-Wan-1.3B.pth  --ref_image "examples/ref1.png,examples/ref2.png" --prompt "暖阳漫过草地，扎着双马尾、头戴绿色蝴蝶结、身穿浅绿色连衣裙的小女孩蹲在盛开的雏菊旁。她身旁一只棕白相间的狗狗吐着舌头，毛茸茸尾巴欢快摇晃。小女孩笑着举起黄红配色、带有蓝色按钮的玩具相机，将和狗狗的欢乐瞬间定格。" --base_seed 42
```

- Multi-GPU inference using FSDP + xDiT USP

``` sh
pip install "xfuser>=0.4.1"
torchrun --nproc_per_node=8 generate.py --task s2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --phantom_ckpt ./Phantom-Wan-1.3B/Phantom-Wan-1.3B.pth  --ref_image "examples/ref3.png,examples/ref4.png" --dit_fsdp --t5_fsdp --ulysses_size 4 --ring_size 2 --prompt "夕阳下，一位有着小麦色肌肤、留着乌黑长发的女人穿上有着大朵立体花朵装饰、肩袖处带有飘逸纱带的红色纱裙，漫步在金色的海滩上，海风轻拂她的长发，画面唯美动人。" --base_seed 42
```

> 💡Note: 
> * Changing `--ref_image` can achieve single reference Subject-to-Video generation or multi-reference Subject-to-Video generation. The number of reference images should be within 4.
> * To achieve the best generation results, we recommend that you describe the visual content of the reference image as accurately as possible when writing `--prompt`. For example, "examples/ref1.png" can be described as "a toy camera in yellow and red with blue buttons".
> * When the generated video is unsatisfactory, the most straightforward solution is to try changing the `--base_seed` and modifying the description in the `--prompt`.

For more inference examples, please refer to "infer.sh". You will get the following generated results:

<table style="width: 100%; border-collapse: collapse; text-align: center; border: 1px solid #ccc;">
  <tr>
    <th style="text-align: center;">
      <strong>Reference Images</strong>
    </th>
    <th style="text-align: center;">
      <strong>Generated Videos</strong>
    </th>
  </tr>

  <tr>
    <td style="text-align: center; vertical-align: middle;">
      <img src="examples/ref1.png" alt="Image 1" style="height: 180px;">
      <img src="examples/ref2.png" alt="Image 2" style="height: 180px;">
    </td>
    <td style="text-align: center; vertical-align: middle;">
      <img src="examples/ref_results/result1.gif" alt="GIF 1" style="width: 400px;">
    </td>
  </tr>

  <tr>
    <td style="text-align: center; vertical-align: middle;">
      <img src="examples/ref3.png" alt="Image 3" style="height: 180px;">
      <img src="examples/ref4.png" alt="Image 4" style="height: 180px;">
    </td>
    <td style="text-align: center; vertical-align: middle;">
      <img src="examples/ref_results/result2.gif" alt="GIF 2" style="width: 400px;">
    </td>
  </tr>

  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;">
      <img src="examples/ref5.png" alt="Image 5" style="height: 180px;">
      <img src="examples/ref6.png" alt="Image 6" style="height: 180px;">
      <img src="examples/ref7.png" alt="Image 7" style="height: 180px;">
    </td>
    <td style="text-align: center; vertical-align: middle;">
      <img src="examples/ref_results/result3.gif" alt="GIF 3" style="width: 400px;">
    </td>
  </tr>

  <tr>
    <td style="text-align: center; vertical-align: middle;">
      <img src="examples/ref8.png" alt="Image 8" style="height: 100px;">
      <img src="examples/ref9.png" alt="Image 9" style="height: 100px;">
      <img src="examples/ref10.png" alt="Image 10" style="height: 100px;">
      <img src="examples/ref11.png" alt="Image 11" style="height: 100px;">
    </td>
    <td style="text-align: center; vertical-align: middle;">
      <img src="examples/ref_results/result4.gif" alt="GIF 4" style="width: 400px;">
    </td>
  </tr>
</table>



## 🆚 Comparative Results
- **Identity Preserving Video Generation**.
![image](./assets/images/id_eval.png)
- **Single Reference Subject-to-Video Generation**.
![image](./assets/images/ip_eval_s.png)
- **Multi-Reference Subject-to-Video Generation**.
![image](./assets/images/ip_eval_m_00.png)

## Acknowledgements
We would like to express our gratitude to the SEED team for their support. Special thanks to Lu Jiang, Haoyuan Guo, Zhibei Ma, and Sen Wang for their assistance with the model and data. In addition, we are also very grateful to Siying Chen, Qingyang Li, and Wei Han for their help with the evaluation.

## BibTeX
```bibtex
@article{liu2025phantom,
  title={Phantom: Subject-Consistent Video Generation via Cross-Modal Alignment},
  author={Liu, Lijie and Ma, Tianxaing and Li, Bingchuan and Chen, Zhuowei and Liu, Jiawei and He, Qian and Wu, Xinglong},
  journal={arXiv preprint arXiv:2502.11079},
  year={2025}
}
```

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=Phantom-video/Phantom&type=Date)](https://www.star-history.com/#Phantom-video/Phantom&Date)