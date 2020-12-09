import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
import logging
import logging.config
_logger = logging.getLogger(__name__)
from . import networks


class BaseModel(ABC):
	""" 本类是 模型类(models) 的抽象基类
	如果要创建子类, 需要实现下面五个函数:
		-- <__init__>:					  初始化类; 首先需要调用 BaseModel.__init__(self, opt).
		-- <set_input>:					 从 dataset 实例中解包并进行数据预处理
		-- <forward>:					  前向计算产生中间结果
		-- <optimize_parameters>:		  计算 loss, gradients,更新 network weights.
		-- <modify_commandline_options>:	(可选项) 增加 model-specific 选项并设置默认选项
	"""

	def __init__(self, opt):
		"""初始化父类 BaseModel

		参数:
			opt (Option class)-- 存储所有实验标志; 需继承自 BaseOption 类

		当创建自己的定制类时，需要自己实现自己的初始化方法
		在本类中, 需要首先调用 BaseModel.__init__(self, opt)
		之后, 需要定义四个列表:
			 self.loss_names (str list):		  指定需要绘制和保存的训练损失
			 self.model_names (str list):		  定义训练中用到的网络模型
			 self.visual_names (str list):		  指定需要显示和保存的图片
			 self.optimizers (optimizer list):	  定义及初始化 optimizers。可以为每个网络定义一个优化器。If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an usage.
		"""
		self.opt = opt
		self.gpu_ids = opt.gpu_ids
		self.isTrain = opt.isTrain
		self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
		self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # 将所有 checkpoints 保存入 save_dir
		if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
			torch.backends.cudnn.benchmark = True
		self.loss_names = []
		self.model_names = []
		self.visual_names = []
		self.optimizers = []
		self.image_paths = []
		self.metric = 0  # 用于学习率策略 'plateau'

	@staticmethod
	def modify_commandline_options(parser, is_train):
		"""加入新的 特定数据集命令行参数（dataset-specific options），并重新指定已知参数选项的值。

		Parameters:
			parser			原始命令行选项解释器
			is_train (bool) -- 训练/测试 阶段。可以使用该标志再增加 特定的训练/测试选项。

		Returns:
			修改后的命令行选项解释器
		"""
		return parser

	@abstractmethod
	def set_input(self, input):
		""" 将 dataloader 中数据解包并进行必要的数据预处理

		Parameters:
			input (dict): 数据与数据信息
		"""
		pass

	@abstractmethod
	def forward(self):
		"""前向运算。被 optimize_parameters 和 test 两个函数调用"""
		pass

	@abstractmethod
	def optimize_parameters(self):
		""" 计算损失，梯度，更新网络权重；每次训练迭代中被调用 """
		pass

	def setup(self, opt):
		""" 加载 & 打印网络；创建 schedulers

		Parameters:
			opt (Option class)-- 存储所有实验标志; 需继承自 BaseOption 类
		"""
		if self.isTrain:
			self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
		if not self.isTrain or opt.continue_train:
			load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
			self.load_networks(load_suffix)
		self.print_networks(opt.verbose)

	def eval(self):
		""" 设置测试时模型为 eval 模式"""
		for name in self.model_names:
			if isinstance(name, str):
				net = getattr(self, 'net' + name)
				net.eval()

	def test(self):
		"""测试时前向函数

		该函数包装了 no_grad() 下的前向函数, 所以我们不需要保存反向时的中间步骤
		它同样调用了 compute_visuals() 来产生额外的可视化结果
		"""
		with torch.no_grad():
			self.forward()
			self.compute_visuals()

	def compute_visuals(self):
		"""为 visdom 和 HTML 可视化类 计算额外的输出图片 """
		pass

	def get_image_paths(self):
		""" 返回用于加载当前数据的图片路径"""
		return self.image_paths

	def update_learning_rate(self):
		""" 对所有网络更新学习率; 每次 epoch 结束被调用 """
		old_lr = self.optimizers[0].param_groups[0]['lr']
		for scheduler in self.schedulers:
			if self.opt.lr_policy == 'plateau':
				scheduler.step(self.metric)
			else:
				scheduler.step()

		lr = self.optimizers[0].param_groups[0]['lr']
		print('learning rate %.7f -> %.7f' % (old_lr, lr))

	def get_current_visuals(self):
		""" 返回可视化图片. train.py 会使用 visdom 展示这些图片, 并保存图片到 HTML 类"""
		visual_ret = OrderedDict()
		for name in self.visual_names:
			if isinstance(name, str):
				visual_ret[name] = getattr(self, name)
		return visual_ret

	def get_current_losses(self):
		""" 返回训练 losses / errors. train.py 会在终端中打印这些 errors, 并保存到文件中"""
		errors_ret = OrderedDict()
		for name in self.loss_names:
			if isinstance(name, str):
				errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
		return errors_ret

	def save_networks(self, epoch):
		"""保存所有网络到硬盘

		Parameters:
			epoch (int) -- 当前 epoch; 在文件 '%s_net_%s.pth' % (epoch, name) 中使用
		"""
		for name in self.model_names:
			if isinstance(name, str):
				save_filename = '%s_net_%s.pth' % (epoch, name)
				save_path = os.path.join(self.save_dir, save_filename)
				net = getattr(self, 'net' + name)

				if len(self.gpu_ids) > 0 and torch.cuda.is_available():
					torch.save(net.module.cpu().state_dict(), save_path)
					net.cuda(self.gpu_ids[0])
				else:
					torch.save(net.cpu().state_dict(), save_path)

	def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
		""" 修复 InstanceNorm checkpoints 兼容性 (prior to 0.4)"""
		key = keys[i]
		if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
			if module.__class__.__name__.startswith('InstanceNorm') and \
					(key == 'running_mean' or key == 'running_var'):
				if getattr(module, key) is None:
					state_dict.pop('.'.join(keys))
			if module.__class__.__name__.startswith('InstanceNorm') and \
			   (key == 'num_batches_tracked'):
				state_dict.pop('.'.join(keys))
		else:
			self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

	def load_networks(self, epoch):
		"""从硬盘加载所有网络

		Parameters:
			epoch (int) -- 当前 epoch; 在文件 '%s_net_%s.pth' % (epoch, name) 中使用
		"""
		_logger.error(" epoch: %s" % epoch)
		for name in self.model_names:
			if isinstance(name, str):
				load_filename = '%s_net_%s.pth' % (epoch, name)
				load_path = os.path.join(self.save_dir, load_filename)
				net = getattr(self, 'net' + name)
				if isinstance(net, torch.nn.DataParallel):
					net = net.module
				print('loading the model from %s' % load_path)
				# if you are using PyTorch newer than 0.4 (e.g., built from
				# GitHub source), you can remove str() on self.device
				state_dict = torch.load(load_path, map_location=str(self.device))
				if hasattr(state_dict, '_metadata'):
					del state_dict._metadata

				# patch InstanceNorm checkpoints prior to 0.4
				for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
					self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
				net.load_state_dict(state_dict)

	def print_networks(self, verbose):
		"""打印网络参数和（如果指定）网络结构

		Parameters:
			verbose (bool) -- 是否打印网络结构
		"""
		print('---------- Networks initialized -------------')
		for name in self.model_names:
			if isinstance(name, str):
				net = getattr(self, 'net' + name)
				num_params = 0
				for param in net.parameters():
					num_params += param.numel()
				if verbose:
					print(net)
				print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
		print('-----------------------------------------------')

	def set_requires_grad(self, nets, requires_grad=False):
		"""对所有网络设置 requies_grad=Fasle 避免不必要的计算
		Parameters:
			nets (network list)   -- 网络列表
			requires_grad (bool)  -- 网络是否需要计算梯度
		"""
		if not isinstance(nets, list):
			nets = [nets]
		for net in nets:
			if net is not None:
				for param in net.parameters():
					param.requires_grad = requires_grad
