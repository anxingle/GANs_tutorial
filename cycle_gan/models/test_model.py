from .base_model import BaseModel
from . import networks


class TestModel(BaseModel):
	""" 本类可以用于生成单向 CycleGAN 结果; 会自动设置 '--dataset_mode single', 这样仅从单向加载图片 """
	@staticmethod
	def modify_commandline_options(parser, is_train=True):
		""" 增加新的 特定数据集命令行参数(dataset-specific options),并重新指定已知参数选项的值

		Parameters:
			parser          原始命令行选项解释器
			is_train (bool) -- 训练/测试 阶段。可以使用该标志再增加 特定的训练/测试选项。

		Returns:
			修改后的命令行选项解释器

		该模型仅可在测试时使用，需要设置 '--dataset_mode single'.
		需要通过 '--model_suffix' 来指定使用的网络
		"""
		assert not is_train, 'TestModel cannot be used during training time'
		parser.set_defaults(no_dropout=True)  # 测试阶段默认没有使用 dropout
		parser.set_defaults(dataset_mode='single')
		parser.add_argument('--model_suffix', type=str, default='', help='在目录 checkpoints_dir 中, [epoch]_net_G[model_suffix].pth 将会作为生成器来加载')

		return parser

	def __init__(self, opt):
		""" 初始化 pix2pix 类

		Parameters:
			opt (Option class)-- 存储所有实验参数; 需要继承自 BaseOptions
		"""
		assert(not opt.isTrain)
		BaseModel.__init__(self, opt)
		# 指定需要打印的训练损失. 训练/测试 脚本会调用 <BaseModel.get_current_losses>
		self.loss_names = []
		# 指定想 保存/显示 的图片. 训练/测试 脚本会调用 <BaseModel.get_current_visuals>
		self.visual_names = ['real', 'fake']
		# 指定需要保存在硬盘上的模型. 训练/测试 脚本会调用  <BaseModel.save_networks>  <BaseModel.load_networks>
		self.model_names = ['G' + opt.model_suffix]  # 仅需要生成器
		self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
									  opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

		# 通过 self.netG_[suffix] 加载模型
		setattr(self, 'netG' + opt.model_suffix, self.netG)

	def set_input(self, input):
		""" 将 dataloader 中的 input_data 解包, 并进行数据前处理

		Parameters:
			input: 包含数据及数据元信息的目录

		需要使用 'single_dataset' 数据集模式. 它仅从一个域加载图片
		"""
		self.real = input['A'].to(self.device)
		self.image_paths = input['A_paths']

	def forward(self):
		self.fake = self.netG(self.real)  # G(real)

	def optimize_parameters(self):
		""" 测试阶段无优化 """
		pass
