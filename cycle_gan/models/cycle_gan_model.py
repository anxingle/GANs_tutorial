import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class CycleGANModel(BaseModel):
	"""
	该类实现 CycleGAN 模型，也就是非成对数据集的 image-to-image 翻译。

	模型训练需要使用 '--dataset_mode unaligned' 数据集实例.
	默认使用 '--netG resnet_9blocks' ResNet 生成器,
	'--netD basic' 判别器 (pix2pix 中引入的 PatchGAN),
	最小平方 GANs 约束 ('--gan_mode lsgan').

	CycleGAN 原文: https://arxiv.org/pdf/1703.10593.pdf
	"""
	@staticmethod
	def modify_commandline_options(parser, is_train=True):
		"""加入新的 特定数据集命令行参数（dataset-specific options），并重新指定已知参数选项的值。

		Parameters:
			parser			原始命令行选项解释器
			is_train (bool) -- 训练/测试 阶段。可以使用该标志再增加 特定的训练/测试选项。

		Returns: 修改后的命令行选项解释器

		CycleGAN中除了 GAN losses, 我们还引入了 lambda_A, lambda_B, 和 lambda_identity 三个权重来加权下面几个损失。
		A (原始域), B (目标域).
		生成器: G_A: A -> B; G_B: B -> A.
		判别器: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
		前向 cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
		反向 cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
		Identity loss (可选): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
		原始 CycleGAN 论文没有使用 dropout。
		"""
		parser.set_defaults(no_dropout=True)  # CycleGAN 默认没有使用 dropout
		if is_train:
			parser.add_argument('--lambda_A', type=float, default=10.0, help='cycle loss (A -> B -> A) 的权重')
			parser.add_argument('--lambda_B', type=float, default=10.0, help='cycle loss (B -> A -> B) 的权重')
			parser.add_argument('--lambda_identity', type=float, default=0.5, help='identity 映射. 将 lambda_identity 设为非 0 可以调整 identity 映射的 loss。举个🌰，如果 identity loss 是 reconstruction loss 的十分之一，则设置 lambda_identity = 0.1')

		return parser

	def __init__(self, opt):
		""" 初始化 CycleGAN 类.

		Parameters:
			opt (Option 类)-- 实验的命令行选项，需继承自 BaseOptions
		"""
		BaseModel.__init__(self, opt)
		# 指定需要打印的训练损失，训练/测试 脚本会调用 BaseModel.get_current_losses 来得到 losses 的值
		self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
		# 指定需要 保存/展示 的图片。 训练/测试 脚本会调用 BaseModel.get_current_visuals 来得到 保存/展示 的图片
		visual_names_A = ['real_A', 'fake_B', 'rec_A']
		visual_names_B = ['real_B', 'fake_A', 'rec_B']
		if self.isTrain and self.opt.lambda_identity > 0.0:  # 如果 identity loss 使用，我们也对 idt_B=G_A(B), idt_A=G_A(B) 两个损失可视化
			visual_names_A.append('idt_B')
			visual_names_B.append('idt_A')

		self.visual_names = visual_names_A + visual_names_B
		# 指定需要保存到硬盘的模型，训练/测试 脚本会调用 BaseModel.save_networks 和 BaseModel.load_networks 两个方法
		if self.isTrain:
			self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
		else:  # 测试阶段仅加载 生成器
			self.model_names = ['G_A', 'G_B']

		# 定义网络：生成器与判别器
		# 这里网络名字和文章中的名字不同。代码（VS 文章）: G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
		self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
										not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
		self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
										not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

		if self.isTrain:  # 定义判别器
			self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
											opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
			self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
											opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

		if self.isTrain:
			if opt.lambda_identity > 0.0:  # 仅当输入与输出的图片有同样通道数的时候才 work(通道数不一样还怎么比较啊！)
				assert(opt.input_nc == opt.output_nc)
			self.fake_A_pool = ImagePool(opt.pool_size)  # 创建图片缓冲池来存储先前生成的图片
			self.fake_B_pool = ImagePool(opt.pool_size)
			# 定义 loss function
			self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # 定义 GAN loss
			self.criterionCycle = torch.nn.L1Loss()
			self.criterionIdt = torch.nn.L1Loss()
			# 初始化 optimizers; 父类 BaseModel.setup 会自动创建 schedulers
			self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
			self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
			self.optimizers.append(self.optimizer_G)
			self.optimizers.append(self.optimizer_D)

	def set_input(self, input):
		""" 将 dataloader 中数据解包并进行数据预处理

		Parameters:
			input (dict): 数据与数据信息
			选项 'direction' 可用于交换 A/B 域
		"""
		AtoB = self.opt.direction == 'AtoB'
		self.real_A = input['A' if AtoB else 'B'].to(self.device)
		self.real_B = input['B' if AtoB else 'A'].to(self.device)
		self.image_paths = input['A_paths' if AtoB else 'B_paths']

	def forward(self):
		"""前向运算。被 optimize_parameters 和 test 两个函数调用"""
		self.fake_B = self.netG_A(self.real_A)  # G_A(A)
		self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
		self.fake_A = self.netG_B(self.real_B)  # G_B(B)
		self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

	def backward_D_basic(self, netD, real, fake):
		""" 计算判别器的 GAN loss

		Parameters:
			netD (network)	  -- 判别器D
			real (tensor array) -- 真实图片
			fake (tensor array) -- 生成器生成的图片

		返回 判别器loss
		这里也调用 loss_D.backward() 来计算梯度
		"""
		# 真实图片
		pred_real = netD(real)
		loss_D_real = self.criterionGAN(pred_real, True)
		# 生成图片
		pred_fake = netD(fake.detach())
		loss_D_fake = self.criterionGAN(pred_fake, False)
		# 结合两个 loss 并计算梯度
		loss_D = (loss_D_real + loss_D_fake) * 0.5
		loss_D.backward()
		return loss_D

	def backward_D_A(self):
		""" 计算 判别器 D_A的 GAN loss"""
		fake_B = self.fake_B_pool.query(self.fake_B)
		self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

	def backward_D_B(self):
		""" 计算 判别器 D_B的 GAN loss"""
		fake_A = self.fake_A_pool.query(self.fake_A)
		self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

	def backward_G(self):
		""" 计算生成器 G_A 和 G_B 的 loss"""
		lambda_idt = self.opt.lambda_identity
		lambda_A = self.opt.lambda_A
		lambda_B = self.opt.lambda_B
		# Identity loss
		if lambda_idt > 0:
			# 如果real_B送入G_A网络的话，G_A 应该和原始 real_B 一模一样。||G_A(B) - B||
			self.idt_A = self.netG_A(self.real_B)
			self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
			# 如果real_A送入G_B网络的话，G_B 应该和原始 real_A 一模一样。||G_B(A) - A||
			self.idt_B = self.netG_B(self.real_A)
			self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
		else:
			self.loss_idt_A = 0
			self.loss_idt_B = 0

		# GAN loss D_A(G_A(A))
		self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
		# GAN loss D_B(G_B(B))
		self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
		# 前向 cycle loss || G_B(G_A(A)) - A||
		self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
		# 反向 cycle loss || G_A(G_B(B)) - B||
		self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
		# 集合 loss 并计算 梯度
		self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
		self.loss_G.backward()

	def optimize_parameters(self):
		""" 计算损失，梯度，更新网络权重；每次训练迭代中被调用 """
		# forward
		self.forward()	  # 计算“假”（生产的）的图片并重建图片
		# G_A and G_B
		self.set_requires_grad([self.netD_A, self.netD_B], False)  # 当更新生成器的时候，判别器不需要更新梯度
		self.optimizer_G.zero_grad()  # 将生成器G_A 和 G_B 梯度清零
		self.backward_G()			 # 将生成器 G_A and G_B 的梯度反向传导
		self.optimizer_G.step()	   # 更新生成器 G_A and G_B 的权重
		# D_A and D_B
		self.set_requires_grad([self.netD_A, self.netD_B], True)
		self.optimizer_D.zero_grad()   # 将判别器 D_A and D_B 的梯度清零
		self.backward_D_A()	  # 对判别器 D_A 计算梯度
		self.backward_D_B()	  # 对判别器 D_B 计算梯度
		self.optimizer_D.step()  # 更新判别器 D_A and D_B 的权重
