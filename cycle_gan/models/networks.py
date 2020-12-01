import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


###############################################################################
# 辅助函数
###############################################################################


class Identity(nn.Module):
	def forward(self, x):
		return x


def get_norm_layer(norm_type='instance'):
	"""返回 normalization 层

	Parameters:
		norm_type (str) -- normalization 层名: batch | instance | none

	XXX: how to understand?
	For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
	For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
	"""
	if norm_type == 'batch':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
	elif norm_type == 'none':
		def norm_layer(x): return Identity()
	else:
		raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
	return norm_layer


def get_scheduler(optimizer, opt):
	""" 返回学习率 scheduler

	Parameters:
		optimizer		  -- 网络优化器
		opt (option class) -- 存储所有实验标志; 需要继承自 BaseOptions．　
							  opt.lr_policy 是学习率策略: linear | step | plateau | cosine

	'linear': 在第一个 opt.n_epochs 轮次内保持相同的学习率，在下个 opt.n_epochs_decay 轮次线性递减到0.
	对于其他 schedulers (step, plateau, and cosine), 我们使用默认的 PyTorch schedulers.
	参阅 https://pytorch.org/docs/stable/optim.html 查看更多细节
	"""
	if opt.lr_policy == 'linear':
		def lambda_rule(epoch):
			lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
			return lr_l
		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
	elif opt.lr_policy == 'step':
		scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
	elif opt.lr_policy == 'plateau':
		scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
	elif opt.lr_policy == 'cosine':
		scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
	else:
		return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
	return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
	""" 初始化网络权重

	Parameters:
		net (network)   -- 需要被初始化的网络
		init_type (str) -- 初始化方法: normal | xavier | kaiming | orthogonal
		init_gain (float)	-- normal, xavier, orthogonal 的缩放因子

	在 CycleGAN和pix2pix原文中使用 'normal'. 可能 xavier 和 kaiming 在其他一些应用中效果更好，可以尝试一下
	"""
	def init_func(m):  # 定义初始化函数
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
			if init_type == 'normal':
				init.normal_(m.weight.data, 0.0, init_gain)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=init_gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif init_type == 'orthogonal':
				init.orthogonal_(m.weight.data, gain=init_gain)
			else:
				raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
			if hasattr(m, 'bias') and m.bias is not None:
				init.constant_(m.bias.data, 0.0)
		elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
			init.normal_(m.weight.data, 1.0, init_gain)
			init.constant_(m.bias.data, 0.0)

	print('initialize network with %s' % init_type)
	net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
	"""初始化网络: 1. 登记 CPU/GPU 设备 (多 GPU 支持); 2. 初始化网络权重
	Parameters:
		net (network)	  -- 需要初始化的网络
		init_type (str)	-- 初始化方法: normal | xavier | kaiming | orthogonal
		gain (float)	   -- normal, xavier, orthogonal 的缩放因子
		gpu_ids (int list) -- 网络在哪个 GPU 上跑: 0,1,2

	返回初始化的网络.
	"""
	if len(gpu_ids) > 0:
		assert(torch.cuda.is_available())
		net.to(gpu_ids[0])
		net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
	init_weights(net, init_type, init_gain=init_gain)
	return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
	"""产生一个生成器

	Parameters:
		input_nc (int) -- 输入图片的通道数
		output_nc (int) -- 输出图片的通道数
		ngf (int) -- 最后一层卷积层的滤波器
		netG (str) -- 生成器名字: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
		norm (str) -- 网络使用的 normalization 层: batch | instance | none
		use_dropout (bool) -- 是否使用 dropout
		init_type (str)	-- 初始化方法
		init_gain (float)  -- normal, xavier and orthogonal 的缩放因子
		gpu_ids (int list) -- 网络运行在哪个GPU上: e.g., 0,1,2

	返回一个生成器

	当前提供两种生成器的实现:
		U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
		The original U-Net paper: https://arxiv.org/abs/1505.04597

		Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
		Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
		We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


	返回的生成器已经被 init_net 初始化了. 使用 RELU 加入非线性部分.
	"""
	net = None
	norm_layer = get_norm_layer(norm_type=norm)

	if netG == 'resnet_9blocks':
		net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
	elif netG == 'resnet_6blocks':
		net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
	elif netG == 'unet_128':
		net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
	elif netG == 'unet_256':
		net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
	else:
		raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
	return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
	"""创建判别器

	Parameters:
		input_nc (int)	 -- 输入图片的通道数
		ndf (int)		  -- 第一个卷积层的滤波器数量
		netD (str)		 -- 判别器结构的名字: basic | n_layers | pixel
		n_layers_D (int)	 -- 判别器中卷积层数量; 当参数 netD=='n_layers' 时此参数有效
		norm (str)		 -- 此网络使用的 normalization 层类型
		init_type (str)	-- 初始化方法
		init_gain (float)  -- normal, xavier and orthogonal 的缩放因子
		gpu_ids (int list) -- 网络运行在哪个GPU上: e.g., 0,1,2

	返回一个判别器

	我们目前提供了三种判别器实现:
		[basic]: 'PatchGAN' 原始 pix2pix 文章描述的分类器
		可以对 70x70 的重叠块判断真假
		这样的 patch-级别的判别器结构比全图级别的判别器有更少量参数，而且可在任意大小的图片上以全卷积的方式工作。

		[n_layers]: 这种模式下，可以用参数 n_layers_D 来指定判别器的卷积层数量（默认=3）

		[pixel]: 1x1 PixelGAN 判别器能够分辨一个像素是否为真。
		It encourages greater color diversity but has no effect on spatial statistics.

	判别器使用 init_net 进行初始化，使用 Leakly RELU 进行非线性初始化
	"""
	net = None
	norm_layer = get_norm_layer(norm_type=norm)

	if netD == 'basic':  # 默认 PatchGAN 分类器
		net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
	elif netD == 'n_layers':
		net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
	elif netD == 'pixel':	 # 对每个像素判别真假
		net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
	else:
		raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
	return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
	"""定义不同的GAN约束

	GANLoss 类抽象了创建和输入相同尺寸的目标标签的需要
	"""

	def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
		""" 初始化 GANLoss 类

		Parameters:
			gan_mode (str) - - GAN目标的类型，它目前支持 vanilla, lsgan, and wgangp.
			target_real_label (bool) - - 真实图片的标签
			target_fake_label (bool) - - 重建图片的标签

		注意: 不用在判别器的最后一层使用 sigmoid. LSGAN 不需要使用 sigmoid; 原始 GANs 使用 BCEWithLogitsLoss
		"""
		super(GANLoss, self).__init__()
		self.register_buffer('real_label', torch.tensor(target_real_label))
		self.register_buffer('fake_label', torch.tensor(target_fake_label))
		self.gan_mode = gan_mode
		if gan_mode == 'lsgan':
			self.loss = nn.MSELoss()
		elif gan_mode == 'vanilla':
			self.loss = nn.BCEWithLogitsLoss()
		elif gan_mode in ['wgangp']:
			self.loss = None
		else:
			raise NotImplementedError('gan mode %s not implemented' % gan_mode)

	def get_target_tensor(self, prediction, target_is_real):
		"""创建和输入同样尺寸的 label

		Parameters:
			prediction (tensor) - - 通常预测来自判别器
			target_is_real (bool) - - ground truth 应为真实图片还是重建图片

		返回:
			目标tensor扩展为 ground truth
		"""

		if target_is_real:
			target_tensor = self.real_label
		else:
			target_tensor = self.fake_label
		return target_tensor.expand_as(prediction)

	def __call__(self, prediction, target_is_real):
		"""计算判别器的输出和 ground truth 的 loss

		Parameters:
			prediction (tensor) - - 通常为判别器的预测输出
			target_is_real (bool) - - ground truth 应为真实图片还是重建图片

		Returns:
			loss
		"""
		if self.gan_mode in ['lsgan', 'vanilla']:
			target_tensor = self.get_target_tensor(prediction, target_is_real)
			loss = self.loss(prediction, target_tensor)
		elif self.gan_mode == 'wgangp':
			if target_is_real:
				loss = -prediction.mean()
			else:
				loss = prediction.mean()
		return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
	"""计算梯度惩罚损失, 在WGAN-GP中使用了 https://arxiv.org/abs/1704.00028

	Arguments:
		netD (network)			  -- 判别器网络
		real_data (tensor array)	-- 真实图片
		fake_data (tensor array)	-- 生成器重建的图片
		device (str)				-- GPU / CPU: torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
		type (str)				  -- 是否将真实图片与重建图片混合: [real | fake | mixed].
		constant (float)			-- 公式 ( ||gradient||_2 - constant)^2 中使用的常量
		lambda_gp (float)		   -- 本loss的权重

	返回梯度惩罚loss
	"""
	if lambda_gp > 0.0:
		if type == 'real':   # 使用真实图片，重建图片还是线性插值的两者
			interpolatesv = real_data
		elif type == 'fake':
			interpolatesv = fake_data
		elif type == 'mixed':
			alpha = torch.rand(real_data.shape[0], 1, device=device)
			alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
			interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
		else:
			raise NotImplementedError('{} not implemented'.format(type))
		interpolatesv.requires_grad_(True)
		disc_interpolates = netD(interpolatesv)
		gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
										grad_outputs=torch.ones(disc_interpolates.size()).to(device),
										create_graph=True, retain_graph=True, only_inputs=True)
		gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
		gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp		# added eps
		return gradient_penalty, gradients
	else:
		return 0.0, None


class ResnetGenerator(nn.Module):
	"""基于 Resnet 的生成器；Resnet 模块在几个 downsampling/upsampling 之间.

	我们借鉴了 Justin Johnson 的风格迁移项目(https://github.com/jcjohnson/fast-neural-style) 代码和思想
	"""

	def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
		""" 构建一个基于 Resnet 的生成器

		Parameters:
			input_nc (int)	  -- 输入图片的通道数
			output_nc (int)	 -- 输出图片的通道数
			ngf (int)		   -- 最后一层卷积层的滤波器数量
			norm_layer		  -- normalization 层
			use_dropout (bool)  -- 是否使用 dropout
			n_blocks (int)	  -- ResNet blocks 的数量
			padding_type (str)  -- 卷积层中 padding的类型: reflect | replicate | zero
		"""
		assert(n_blocks >= 0)
		super(ResnetGenerator, self).__init__()
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		model = [nn.ReflectionPad2d(3),
				 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
				 norm_layer(ngf),
				 nn.ReLU(True)]

		n_downsampling = 2
		for i in range(n_downsampling):  # 增加 downsampling 层
			mult = 2 ** i
			model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
					  norm_layer(ngf * mult * 2),
					  nn.ReLU(True)]

		mult = 2 ** n_downsampling
		for i in range(n_blocks):	   # 增加 ResNet 模块

			model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

		for i in range(n_downsampling):  # 增加 upsampling 层
			mult = 2 ** (n_downsampling - i)
			model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
										 kernel_size=3, stride=2,
										 padding=1, output_padding=1,
										 bias=use_bias),
					  norm_layer(int(ngf * mult / 2)),
					  nn.ReLU(True)]
		model += [nn.ReflectionPad2d(3)]
		model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
		model += [nn.Tanh()]

		self.model = nn.Sequential(*model)

	def forward(self, input):
		return self.model(input)


class ResnetBlock(nn.Module):
	"""定义 Resnet 模块"""

	def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
		"""初始化 Resnet 模块

		Resnet 模块是使用了 skip 连接的卷积模块
		使用 build_conv_block 函数构建卷积模块，并在 forward 函数中实现了 skip 连接
		Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
		"""
		super(ResnetBlock, self).__init__()
		self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

	def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
		"""构建卷积模块

		Parameters:
			dim (int)		   -- 卷积层的通道数
			padding_type (str)  -- padding层: reflect | replicate | zero
			norm_layer		  -- normalization 层
			use_dropout (bool)  -- 是否使用 dropout
			use_bias (bool)	 -- 卷积层是否使用 bias

		返回 卷积模块 (conv layer, normalization layer, non-linearity layer (ReLU))
		"""
		conv_block = []
		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)

		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
		if use_dropout:
			conv_block += [nn.Dropout(0.5)]

		p = 0
		if padding_type == 'reflect':
			conv_block += [nn.ReflectionPad2d(1)]
		elif padding_type == 'replicate':
			conv_block += [nn.ReplicationPad2d(1)]
		elif padding_type == 'zero':
			p = 1
		else:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)
		conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

		return nn.Sequential(*conv_block)

	def forward(self, x):
		out = x + self.conv_block(x)  # skip connections
		return out


class UnetGenerator(nn.Module):
	"""创建基于 Unet 的生成器"""

	def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
		"""构建 Unet 生成器
		Parameters:
			input_nc (int)  -- 输入图片的通道
			output_nc (int) -- 输出图片的通道
			num_downs (int) -- UNet 下采样的数量. 举例，如果|num_downs| == 7, 128x128的图片会变成 1x1
			ngf (int)   --  最后一层卷积层的滤波器数量
			norm_layer  --  normalization层

		从里到外构建 U-Net，这是个递归过程
		"""
		super(UnetGenerator, self).__init__()
		# 构建 UNet 结构
		unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
		for i in range(num_downs - 5):		  # 增加中间层 ngf * 8个滤波器
			unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
		# 逐渐将滤波器数量从 ngf*8 降到 ngf
		unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
		unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
		unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
		self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

	def forward(self, input):
		return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
	"""使用 skip 连接定义 UNet 子模块
		X -------------------identity----------------------
		|-- 下采样 -- |子模块| -- 上采样 --|
	"""

	def __init__(self, outer_nc, inner_nc, input_nc=None,
				 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
		""" 使用skip 连接定义 UNet 子模块

		Parameters:
			outer_nc (int) -- 外(最顶)层卷积层的滤波器数量
			inner_nc (int) -- 内层卷积层的滤波器数量
			input_nc (int) -- 输入 图片/特征图 的通道数
			submodule (UnetSkipConnectionBlock) -- 上个被定义的子模块
			outermost (bool)	-- 本子模块是否最外层（最后）模块
			innermost (bool)	-- 本子模块是否最内层（最前）模块
			norm_layer		  -- normalization 层
			use_dropout (bool)  -- 是否使用 dropout layers.
		"""
		super(UnetSkipConnectionBlock, self).__init__()
		self.outermost = outermost
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d
		if input_nc is None:
			input_nc = outer_nc
		downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
							 stride=2, padding=1, bias=use_bias)
		downrelu = nn.LeakyReLU(0.2, True)
		downnorm = norm_layer(inner_nc)

		uprelu = nn.ReLU(True)
		upnorm = norm_layer(outer_nc)

		if outermost:
			upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
										kernel_size=4, stride=2,
										padding=1)
			down = [downconv]
			up = [uprelu, upconv, nn.Tanh()]
			model = down + [submodule] + up
		elif innermost:
			upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
										kernel_size=4, stride=2,
										padding=1, bias=use_bias)
			down = [downrelu, downconv]
			up = [uprelu, upconv, upnorm]
			model = down + up
		else:
			upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
										kernel_size=4, stride=2,
										padding=1, bias=use_bias)
			down = [downrelu, downconv, downnorm]
			up = [uprelu, upconv, upnorm]

			if use_dropout:
				model = down + [submodule] + up + [nn.Dropout(0.5)]
			else:
				model = down + [submodule] + up

		self.model = nn.Sequential(*model)

	def forward(self, x):
		if self.outermost:
			return self.model(x)
		else:   # add skip connections
			return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
	"""定义 PatchGAN 判别器"""

	def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
		"""构建 PatchGAN 判别器

		Parameters:
			input_nc (int)  -- 输入图片的通道
			ndf (int)	   -- 最后一层卷积层的滤波器数量
			n_layers (int)  -- 判别器中卷积层数量
			norm_layer	  -- normalization层
		"""
		super(NLayerDiscriminator, self).__init__()
		if type(norm_layer) == functools.partial:  # 没必要使用偏置(BatchNorm2d 有仿射参数)
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		kw = 4
		padw = 1
		sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
		nf_mult = 1
		nf_mult_prev = 1
		for n in range(1, n_layers):  # 逐渐增加卷积核数量
			nf_mult_prev = nf_mult
			nf_mult = min(2 ** n, 8)
			sequence += [
				nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
				norm_layer(ndf * nf_mult),
				nn.LeakyReLU(0.2, True)
			]

		nf_mult_prev = nf_mult
		nf_mult = min(2 ** n_layers, 8)
		sequence += [
			nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
			norm_layer(ndf * nf_mult),
			nn.LeakyReLU(0.2, True)
		]

		sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # 输出为1通道预测特征图
		self.model = nn.Sequential(*sequence)

	def forward(self, input):
		"""Standard forward."""
		return self.model(input)


class PixelDiscriminator(nn.Module):
	"""定义 1x1 PatchGAN 判别器 (pixelGAN)"""

	def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
		"""构造 1x1 PatchGAN 判别器

		Parameters:
			input_nc (int)  -- 输入图片的通道数
			ndf (int)	   -- 最后一层卷积层的滤波器数量
			norm_layer	  -- normalization层
		"""
		super(PixelDiscriminator, self).__init__()
		if type(norm_layer) == functools.partial:  # 没必要使用偏置(BatchNorm2d 有仿射参数)
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		self.net = [
			nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
			norm_layer(ndf * 2),
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

		self.net = nn.Sequential(*self.net)

	def forward(self, input):
		return self.net(input)
