# RePaint-NeRF
Official Implementation for "RePaint-NeRF: NeRF Editting via Semantic Masks and Diffusion Models".



https://github.com/StarsTesla/RePaint-NeRF/assets/45264423/684847b5-025f-49b6-a524-9bdeb1e2c55a


https://github.com/StarsTesla/RePaint-NeRF/assets/45264423/79170e42-11a6-431e-8871-e9f71d96292e



https://github.com/StarsTesla/RePaint-NeRF/assets/45264423/9e41164b-e2c2-4a29-8483-37dd79d045d6



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
editing tasks. See
[Websites](https://repaintnerf.github.io) for a better view into our edited results.

## Getting Starte
1. clone this repo
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
   
coming soon.........

6. About the feature extraction
If you have the need, we are happy to help you with that, while it is kind of hard to install (it use LSeg) which is a different env.
We strongly recommend you refer to [https://github.com/pfnet-research/distilled-feature-fields](https://github.com/pfnet-research/distilled-feature-fields)


## Acknowledgments
This repo is heavily rely on [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion), which is a pytorch implementation of the text-to-3D model Dreamfusion.

## Citation

```
@inproceedings{RePaint-NeRF,
      title={{RePaint-NeRF}: NeRF Editting via Semantic Masks and Diffusion Models}, 
      author={Zhou, Xingchen and He, Ying and Yu, F Richard and Li, Jianqiang and Li, You},
      year={2023},
      booktitle={IJCAI},
```
