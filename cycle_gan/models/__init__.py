"""该部分模块包含了与目标函数，优化器，网络结构相关的模块

要增加一个称为 'dummy' 的模型类, 需要增加文件 'dummy_model.py' 并定义继承自 BaseModel 的子类 DummyModel。
你需要实现下述五个函数:
	-- <__init__>:					  初始化类，首先需调用 BaseModel.__init__(self, opt)
	-- <set_input>:					 从 dataset 实例中解包并进行数据预处理
	-- <forward>:					 前向计算产生中间结果
	-- <optimize_parameters>:		 计算 loss, gradients,更新 network weights.
	-- <modify_commandline_options>:	(可选项) 增加 model-specific 选项并设置默认选项

在函数 <__init__> 中, 需要定义四个列表:
	-- self.loss_names (str list):		  指定需要绘制和保存的训练损失
	-- self.model_names (str list):		  定义训练中用到的网络模型
	-- self.visual_names (str list):		  指定需要显示和保存的图片
	-- self.optimizers (optimizer list):	  定义及初始化 optimizers。可以为每个网络定义一个优化器。If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an usage.

可以通过指定 '--model dummy' 选项来使用模型类，参阅模板类 'template_model.py' 查看细节
"""

import importlib
from models.base_model import BaseModel


def find_model_using_name(model_name):
	"""引入模块 "models/[model_name]_model.py".

	在文件中，类 DatasetNameModel() 会被实例化。它需要是 BaseModel 的子类
	"""
	model_filename = "models." + model_name + "_model"
	modellib = importlib.import_module(model_filename)
	model = None
	target_model_name = model_name.replace('_', '') + 'model'
	for name, cls in modellib.__dict__.items():
		if name.lower() == target_model_name.lower() \
		   and issubclass(cls, BaseModel):
			model = cls

	if model is None:
		print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
		exit(0)

	return model


def get_option_setter(model_name):
	""" 返回 类静态方法 modify_commandline_options """
	model_class = find_model_using_name(model_name)
	return model_class.modify_commandline_options


def create_model(opt):
	""" 根据给定的选项创建 模型实例

	该函数封装了 model 类.
	这是本模块和 'train.py'/'test.py' 的主要接口。

	Example:
		>>> from models import create_model
		>>> model = create_model(opt)
	"""
	model = find_model_using_name(opt.model)
	instance = model(opt)
	print("model [%s] was created" % type(instance).__name__)
	return instance
