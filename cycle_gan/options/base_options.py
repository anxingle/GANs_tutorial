import argparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
	"""定义了训练和测试时的参数

	同样实现了一些诸如 parsing, printing, saving 之类的辅助函数
	收集了在 dataset/model 类的 modify_commandline_options 函数中定义的额外参数选项functions in both dataset class and model class.
	"""

	def __init__(self):
		""" 重置该类; 表示该类还未被初始化 """
		self.initialized = False

	def initialize(self, parser):
		""" 定义 训练/测试中 常见选项 """
		# basic parameters
		parser.add_argument('--dataroot', help='图片训练集路径 (应还有 trainA, trainB, valA, valB 等子文件夹)')
		parser.add_argument('--name', type=str, default='experiment_name', help='实验名称. 决定了将样本和模型存储在哪')
		parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
		parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='模型存储位置')
		# model parameters
		parser.add_argument('--model', type=str, default='cycle_gan', help='选择使用哪个模型 [cycle_gan | pix2pix | test | colorization]')
		parser.add_argument('--input_nc', type=int, default=3, help='# 输入图片通道数: 3 for RGB and 1 for grayscale')
		parser.add_argument('--output_nc', type=int, default=3, help='# 输出图片通道数: 3 for RGB and 1 for grayscale')
		parser.add_argument('--ngf', type=int, default=64, help='# 生成器最后一层卷积层滤波器数量')
		parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
		parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
		parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
		parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
		parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
		parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
		parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
		parser.add_argument('--no_dropout', action='store_true', help='生成器不使用dropout')
		# dataset parameters
		parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
		parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
		parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
		parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
		parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
		parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
		parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
		parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
		parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
		parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
		parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
		# additional parameters
		parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
		parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
		parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
		parser.add_argument('--dicom', action='store_true', help='if specified, print more debugging information')
		parser.add_argument('--norm_max', type=int, default=255, help='dataset min/max values')
		parser.add_argument('--norm_min', type=int, default=0, help='dataset min/max values')
		parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
		self.initialized = True
		return parser

	def gather_options(self, infer=False):
		"""使用基础选项参数初始化 parser(仅一次).
		增加额外的 model-specific 和 dataset-specific 选项
		这些选项定义在 dataset 和 model 类的 <modify_commandline_options> 函数中
		"""
		if not self.initialized:  # 检查是否已被初始化
			parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
			parser = self.initialize(parser)

		if infer:
			parser.add_argument('--port', required=True, type=str, help='推理时后端服务开放端口')
			parser.set_defaults(dataroot='./')  # 推理时不需要指定--dataroot
			parser.set_defaults(norm_max=255)  # 读取非dicom时不需要指定 norm_max
			parser.set_defaults(norm_min=0)  # 读取非dicom时不需要指定 norm_min
			parser.set_defaults(model='test')  # 读取非dicom时不需要指定 norm_min
			parser.set_defaults(dataset_mode='single')  # 推理时数据集类应为单向
		# 得到基本选项
		opt, _ = parser.parse_known_args()

		# 修改 model-related parser 选项
		model_name = opt.model
		model_option_setter = models.get_option_setter(model_name)
		parser = model_option_setter(parser, self.isTrain)
		opt, _ = parser.parse_known_args()  # parse again with new defaults

		# modify dataset-related parser options
		dataset_name = opt.dataset_mode
		dataset_option_setter = data.get_option_setter(dataset_name)
		parser = dataset_option_setter(parser, self.isTrain)

		# 保存并返回 parser
		self.parser = parser
		return parser.parse_args()

	def print_options(self, opt):
		""" 打印并保存 options

		打印当前选项及默认选项值(如果不同).
		将 选项 保存入文本文件 / [checkpoints_dir] / opt.txt
		"""
		message = ''
		message += '----------------- Options ---------------\n'
		for k, v in sorted(vars(opt).items()):
			comment = ''
			default = self.parser.get_default(k)
			if v != default:
				comment = '\t[default: %s]' % str(default)
			message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
		message += '----------------- End -------------------'
		print(message)

		# save to the disk
		expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
		util.mkdirs(expr_dir)
		file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
		with open(file_name, 'wt') as opt_file:
			opt_file.write(message)
			opt_file.write('\n')

	def parse(self, infer=False):
		""" 解析选项, 创建 checkpoints 目录前缀, 设置GPU设备 """
		opt = self.gather_options(infer)
		opt.isTrain = self.isTrain   # train or test

		# process opt.suffix
		if opt.suffix:
			suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
			opt.name = opt.name + suffix

		self.print_options(opt)

		# set gpu ids
		str_ids = opt.gpu_ids.split(',')
		opt.gpu_ids = []
		for str_id in str_ids:
			id = int(str_id)
			if id >= 0:
				opt.gpu_ids.append(id)
		if len(opt.gpu_ids) > 0:
			torch.cuda.set_device(opt.gpu_ids[0])

		self.opt = opt
		return self.opt
