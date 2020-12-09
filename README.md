# GANs tutorial
- [x] cycleGAN
- [ ] pix2pix
- [ ] DCGAN
- [ ] InfoGAN

## cycleGAN

 改写自作者原工程 [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) 。原工程包含 [pix2pix](https://arxiv.org/pdf/1611.07004.pdf) ,同时 dataset 类过于冗余，这里进行裁剪。

- [x] 增加读取 **dicom** 的接口。
- [x] 增加前端展示页面，方便推理使用

###  推理使用
```
python app.py --dataroot ./ --model test -model_suffix _A --norm_max 255 --norm_min 0 --dataset_mode single --gpu_ids -1 --name xrays --netG unet_256
# --norm_max --norm_min 均仅在读取dicom时才会被使用，不用特别关注。详见 unaligned_dataset.py 中 __getitem__() 函数
# --name 为训练的网络名称，在 checkpoints中显示为某目录
```

![](https://raw.githubusercontent.com/anxingle/GANs_tutorial/main/imgs/infer.jpg)

