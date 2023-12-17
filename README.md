# RePaint-NeRF: NeRF Editting via Semantic Masks and Diffusion Models
Official Implementation for "RePaint-NeRF: NeRF Editting via Semantic Masks and Diffusion Models".

https://github.com/StarsTesla/RePaint-NeRF/assets/45264423/2bbb3246-c562-472e-8bb7-c9f8f92aa38a


The emergence of Neural Radiance Fields (NeRF) has promoted the development of synthesized high-fidelity views of the intricate real world. However, it is still a very demanding task to repaint the content in NeRF. In this paper, we propose a novel framework that can take RGB images as input and alter the 3D content in neural scenes. Our work
leverages existing diffusion models to guide changes in the designated 3D content.
Specifically, we semantically select
the object we want to modify first, and a pre-trained diffusion model will guide the NeRF
model to generate new 3D
objects, which can improve the editability, diversity, and application range of NeRF.
Experiment results show that our
algorithm is effective for editing 3D objects in NeRF under different text prompts,
including editing appearance, shape,
etc. We validate our method on real-world datasets and synthetic-world datasets for these
editing tasks. See our
[Websites](https://starstesla.github.io/repaintnerf/) for a better view into our edited results.

**Note Just a pre-document, wating for full testing, sorry..., feel free to use it.**

## Getting Started
1. Clone this repo
```
git clone https://github.com/StarsTesla/RePaint-NeRF
cd RePaint-NeRF
```
2. Install
**Note: We use pytorch 1.31.1 and CUDA 11.7.**
To install the env, you should install Anaconda first, login with huggingface-cli login in command line.
```
conda create -n repaint-nerf python=3.9
conda activate repaint-nerf

pip install -r requirements.txt
```

3. Extensions install
```
cd stable-dreamfusion

# install all extension modules
bash scripts/install_ext.sh

# if you want to install manually, here is an example:
pip install ./raymarching # install to python path (you still need the raymarching/ folder, since this only installs the built extension.)
```
4. Our Data

We prepared the mask data of `hotdog` and `flower` in [google-drive](https://drive.google.com/drive/folders/1x_PIk2nOqA5ywymYiwKdn4rjB-C4CLhj?usp=sharing)
We also encourage you use SAM to get the mask you want, we found that it is a very useful tools for open world and you should just make sure the masks are 3D consistent.

5. About the feature extraction
If you have the need, we are happy to help you with that, while it is kind of hard to install (it use LSeg) which is a different env.
We strongly recommend you refer to [https://github.com/pfnet-research/distilled-feature-fields](https://github.com/pfnet-research/distilled-feature-fields)

## Usage
```
for pretrain
python main.py --text "a cup of coffe on a plate" --text_bg "plate" --workspace hotdog --bound=2.0 --iters=3000 --exp_name=hotdog_base --data_type=blender --data_dir=data --img_wh=[400,400]
for editing
python main.py --text "banana on a plate" --text_bg "plate" --workspace hotdog --bound=2.0 --iters=15000 --exp_name=hotdog_base --data_type=blender --data_dir=data --img_wh=[400,400] --pretrained
```

## Acknowledgments
This repo is heavily rely on [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion), which is a pytorch implementation of the text-to-3D model Dreamfusion.

## Citation
If you find this useful for your research, please cite the following paper.

```
@inproceedings{ijcai2023p201,
  title     = {RePaint-NeRF: NeRF Editting via Semantic Masks and Diffusion Models},
  author    = {Zhou, Xingchen and He, Ying and Yu, F. Richard and Li, Jianqiang and Li, You},
  booktitle = {Proceedings of the Thirty-Second International Joint Conference on
               Artificial Intelligence, {IJCAI-23}},
  pages     = {1813--1821},
  year      = {2023},
  month     = {8}
}
```
