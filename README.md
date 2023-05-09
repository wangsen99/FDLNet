# Boosting Night-time Scene Parsing with <br /> Learnable Frequency

This repo is the official implementation of ["Boosting Night-time Scene Parsing with Learnable Frequency (IEEE TIP 2023)
"](https://ieeexplore.ieee.org/document/10105211).

## Installation

Our work is based on ["
awesome-semantic-segmentation-pytorch"](https://github.com/Tramac/awesome-semantic-segmentation-pytorch), please follow their [README.md](https://github.com/Tramac/awesome-semantic-segmentation-pytorch#readme) for installation.

## Data Preparation

["NightCity"](https://dmcv.sjtu.edu.cn/people/phd/tanxin/NightCity/index.html)

["NightCity+"](https://drive.google.com/file/d/1EDhWx-fcS7pIIBGbu3TpebNrmyE08KzC/view) (Only reannotated val set)

["BDD100K-night"](https://drive.google.com/file/d/1l4Mh3V7OcCbD6GpxPzovloLlRWSAZ4vZ/view?usp=share_link) (Only images, please download the labels from [here](https://bdd-data.berkeley.edu/) with permission)

## Train
```
cd scripts
python train_edge.py --model fdlnet --backbone resnet50 --dataset night --aux
```

## Test
```
cd scripts
python eval.py --model fdlnet --backbone resnet101 --dataset night --aux
```

## Results and Models

| Dataset | mIoU | w/ ms | Model |
| :---: | :---: | :---: | :---: |
| NightCity | 54.60  | 55.42 | [FDLNet (DeeplabV3+)](https://drive.google.com/file/d/15gZHRTOHeasemjv7-GW_Ooxk7m96ZIO2/view?usp=sharing) |
| NightCity+ | 56.20 | 56.79 | ~|

## Citation
If you find this repo useful for your research, please consider citing our paper:
```
@ARTICLE{10105211,
  author={Xie, Zhifeng and Wang, Sen and Xu, Ke and Zhang, Zhizhong and Tan, Xin and Xie, Yuan and Ma, Lizhuang},
  journal={IEEE Transactions on Image Processing}, 
  title={Boosting Night-Time Scene Parsing With Learnable Frequency}, 
  year={2023},
  volume={32},
  pages={2386-2398},
  doi={10.1109/TIP.2023.3267044}}
```
