"""Image-to-image 通用测试脚本。

使用 `train.py` 训练完模型后，使用本脚本来测试模型。它从 `--checkpoints_dir` 目录加载保存的模型，并将推理结果保存入目录 `--results_dir`。

首先从 option 参数中创建 model 实例和 dataset 实例。它将会 **硬编码** 一些参数。之后测试 `--num_test` 张图片并将结果保存入 HTML。

使用示例（需要首先训练模型或者下载预训练模型）:
	测试 CycleGAN model (两边都测):
		python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

	测试 CycleGAN model (仅测一边):
		python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

	选项 '--model test' 用来生成一边的 CycleGAN 结果（比如A-to-B）。
	该选项自动设置 '--dataset_mode single', 这样会自动从一个数据集来加载图片。
	相对来说 '--model cycle_gan' 需要从两个方向来加载图片和生成结果。很多时候不是必要的。
	结果自动保存在 './results/'.
	使用选项 '--results_dir <directory_path_to_save_result>' 来显式指定结果存放目录。

	测试 pix2pix 模型:
		python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

参阅 options/base_options.py and options/test_options.py 了解更多测试参数选项说明.
在这里查看更多训练和测试指南: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
常见问题列表: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import htmls


if __name__ == '__main__':
	opt = TestOptions().parse()
	# 直接指定一些选项进行测试
	opt.num_threads = 0   # 测试时 DataLoader 仅支持 num_threads = 0
	opt.batch_size = 1	# 测试时batch_size设为1
	opt.serial_batches = True  # 关闭数据 shuffling; 需要随机选择图片时再打开
	opt.no_flip = True	# no flip; 如果需要翻转图片，注释本行
	opt.display_id = -1   # no visdom display; 测试时将结果保存入 HTML 文件
	dataset = create_dataset(opt)  # 根据 opt.dataset_mode 和其他选项创建 dataset 实例
	model = create_model(opt)	  # 根据 opt.model 和其他选项创建 model 实例
	model.setup(opt)			   # 常规设置: 加载和打印网络; 创建 schedulers
	# 创建前端展示页面
	web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # 定义前端目录
	if opt.load_iter > 0:  # load_iter 默认为 0
		web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
	print('creating web directory', web_dir)
	webpage = htmls.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
	# 使用 eval 模式测试，这仅影响诸如 batchnorm 和 dropout 层。
	# [pix2pix]: 在原始 pix2pix 中使用 batchnorm 和 dropout。可以试试使用和不使用 eval() 模式的区别。
	# [CycleGAN]: 不会影响CycleGAN。因为CycleGAN 使用 instancenorm，没有使用 dropout。
	if opt.eval:
		model.eval()
	for i, data in enumerate(dataset):
		if i >= opt.num_test:  # 仅进行 opt.num_test 张图片
			break
		model.set_input(data)  # 从 data loader 中得到数据
		model.test()
		visuals = model.get_current_visuals()  # 得到图片结果
		img_path = model.get_image_paths()
		if i % 5 == 0:  # 将图片保存到 HTML 文件
			print('processing (%04d)-th image... %s' % (i, img_path))
		save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
	webpage.save()  # 保存前端页面
