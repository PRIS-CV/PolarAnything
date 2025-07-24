<h2 align="center">PolarAnything: Diffusion-based Polarimetric Image Synthesis</h2>
<h4 align="center">
     <a href="https://scholar.google.com/citations?user=cXbfQI0AAAAJ&hl=zh-CN&authuser=1"><strong>Kailong Zhang<sup>†</sup></strong></a>
    ·
    <a href="https://youweilyu.github.io/"><strong>Youwei Lyu<sup>†</sup></strong></a>
    ·
    <a href="https://gh-home.github.io/"><strong>Heng Guo<sup>*</sup></strong></a>
    ·
    <a href="https://teacher.bupt.edu.cn/lisi/zh_CN/index.htm"><strong>Si Li</strong></a>
    ·
    <a href="https://zhanyuma.cn/"><strong>Zhanyu Ma</strong></a>
    ·
    <a href="https://camera.pku.edu.cn/"><strong>Boxin Shi</strong></a>
</h3>
<h4 align="center"><a href="https://iccv.thecvf.com/">ICCV 2025</a></h3>
<p align="center">
  <br>
    <a href="">
      <img src='https://img.shields.io/badge/arXiv-Paper-981E32?style=for-the-badge&Color=B31B1B' alt='arXiv PDF'>
    </a>
    <a href=''>
      <img src='https://img.shields.io/badge/PolarAnything-Project Page-5468FF?style=for-the-badge' alt='Project Page'></a>
</p>
<div align="center">
</div>


<img src=assets/Teasor.png width=100% />

# Requirements

We test our codes under the following environment: `Ubuntu 22.04, Python 3.9.23, CUDA 12.1`.

1. Clone this repository.

```bash
git clone https://github.com/PRIS-CV/PolarAnything.git
cd PolarAnything
```

2. Install packages

```bash
conda env create -f environment.yaml
```

# Dataset  

The data is still being organized. Coming soon!

# Pre-trained models

We provide the [pre-trained](https://drive.google.com/file/d/1rhgBcNgxhXupAxH0p2DpLftjEVJ4fDIR/view?usp=sharing) models for inference. Just download and put them into the `model` folder.

# Inference

- You can run inference on the example raw images using the following command:

```bash
./run_infer.sh
```

the results will be saved in the `results/` directory

- If you want to use your own data, simply call the Python script directly with your desired parameters. For example:

```bash
python infer.py \
  --input_folder <your_input_folder> \
  --results_folder <your_results_folder> \
```

# Train  

```bash
./run_train.sh
```

# Citation

If you find this work helpful to your research, please cite:

```
@misc{zhang2025polaranythingdiffusionbasedpolarimetricimage,
      title={PolarAnything: Diffusion-based Polarimetric Image Synthesis}, 
      author={Kailong Zhang and Youwei Lyu and Heng Guo and Si Li and Zhanyu Ma and Boxin Shi},
      year={2025},
      eprint={2507.17268},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.17268}, 
}
```

# Acknowledgements

This code is built on [Stable Diffusion](https://stability.ai/). We thank the authors for sharing their codes.
