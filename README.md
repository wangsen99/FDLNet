# Boosting Night-time Scene Parsing with Learnable Frequency

This repo is the official implementation of ["Boosting Night-time Scene Parsing with Learnable Frequency
"](https://arxiv.org/abs/2208.14241)

## Installation

Our work is based on ["
awesome-semantic-segmentation-pytorch"](https://github.com/Tramac/awesome-semantic-segmentation-pytorch), please follow their [README.md](https://github.com/Tramac/awesome-semantic-segmentation-pytorch#readme) for installation.

## Data Preparation

["NightCity"](https://dmcv.sjtu.edu.cn/people/phd/tanxin/NightCity/index.html)

["NightCity+"](https://drive.google.com/file/d/1EDhWx-fcS7pIIBGbu3TpebNrmyE08KzC/view) (Only reannotated val set)

## Results and Models

| Dataset | mIoU | w/ ms | Model |
| :---: | :---: | :---: | :---: |
| NightCity | 54.60  | 55.42 | [FDLNet (DeeplabV3+)](https://drive.google.com/file/d/15gZHRTOHeasemjv7-GW_Ooxk7m96ZIO2/view?usp=sharing) |
| NightCity+ | 56.20 | 56.79 | ~|

### Test
```
cd scripts
python eval.py --model fdlnet --backbone resnet101 --dataset night --aux
```

## Citation
If you find this repo useful for your research, please consider citing our paper:
```
@misc{https://doi.org/10.48550/arxiv.2208.14241,
  doi = {10.48550/ARXIV.2208.14241},
  url = {https://arxiv.org/abs/2208.14241},
  author = {Xie, Zhifeng and Wang, Sen and Xu, Ke and Zhang, Zhizhong and Tan, Xin and Xie, Yuan and Ma, Lizhuang},
  title = {Boosting Night-time Scene Parsing with Learnable Frequency},
  publisher = {arXiv},
  year = {2022},
}
```
