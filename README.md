# GANs tutorial
- [x] cycleGAN
- [ ] pix2pix
- [ ] DCGAN
- [ ] InfoGAN

## cycleGAN

[cycleGAN](https://arxiv.org/abs/1703.10593) 改写自作者原工程 [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) 。原工程包含 [pix2pix](https://arxiv.org/pdf/1611.07004.pdf) ,同时 dataset 类过于冗余，这里进行裁剪，并增加读取 dicom 的接口。