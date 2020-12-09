"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
	-- <__init__>:					  initialize the class, first call BaseDataset.__init__(self, opt).
	-- <__len__>:					   return the size of dataset.
	-- <__getitem__>:				   get a data point from data loader.
	-- <modify_commandline_options>:	(optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
	"""Import the module "data/[dataset_name]_dataset.py".

	In the file, the class called DatasetNameDataset() will
	be instantiated. It has to be a subclass of BaseDataset,
	and it is case-insensitive.
	"""
	dataset_filename = "data." + dataset_name + "_dataset"
	datasetlib = importlib.import_module(dataset_filename)

	dataset = None
	target_dataset_name = dataset_name.replace('_', '') + 'dataset'
	for name, cls in datasetlib.__dict__.items():
		if name.lower() == target_dataset_name.lower() \
		   and issubclass(cls, BaseDataset):
			dataset = cls

	if dataset is None:
		raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

	return dataset


def get_option_setter(dataset_name):
	"""Return the static method <modify_commandline_options> of the dataset class."""
	dataset_class = find_dataset_using_name(dataset_name)
	return dataset_class.modify_commandline_options


def create_dataset(opt):
	"""根据命令行参数创建 dataset 类

	本函数包装 CustomDatasetDataLoader 类
		这是本模块和 'train.py '/'test.py'/'app.py' 的主要接口

	Example:
		>>> from data import create_dataset
		>>> dataset = create_dataset(opt)
	"""
	data_loader = CustomDatasetDataLoader(opt)
	dataset = data_loader.load_data()
	return dataset


class CustomDatasetDataLoader():
	""" 将 DataSet 类包装，并进行多进程加载 """

	def __init__(self, opt):
		"""
		步骤 1: 根据参数 [opt.dataset_mode] 创建 dataset 类实例: unaligned | aligned | single | colorization
		步骤 2: 创建多进程数据加载器
		"""
		self.opt = opt
		dataset_class = find_dataset_using_name(opt.dataset_mode)
		self.dataset = dataset_class(opt)
		print("dataset [%s] was created" % type(self.dataset).__name__)
		self.dataloader = torch.utils.data.DataLoader(
			self.dataset,
			batch_size=opt.batch_size,
			shuffle=not opt.serial_batches,
			num_workers=int(opt.num_threads))

	def load_data(self):
		return self

	def __len__(self):
		"""Return the number of data in the dataset"""
		return min(len(self.dataset), self.opt.max_dataset_size)

	def __iter__(self):
		"""Return a batch of data"""
		for i, data in enumerate(self.dataloader):
			if i * self.opt.batch_size >= self.opt.max_dataset_size:
				break
			yield data
