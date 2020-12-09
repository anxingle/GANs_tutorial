import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class UnalignedDataset(BaseDataset):
	"""
	该类可以加载 非对齐/非成对 数据集

	它需要两个目录来加载训练数据集: domain A '/path/to/data/trainA'; domain B '/path/to/data/trainB'.
	可以使用标志 '--dataroot /path/to/data' 来训练模型
	同样，测试时也需要准备好两个目录: '/path/to/data/testA', '/path/to/data/testB'
	"""

	def __init__(self, opt):
		""" 初始化 dataset 类

		Parameters:
			opt (Option class) -- 存储所有实验参数; 需要继承自 BaseOptions
		"""
		BaseDataset.__init__(self, opt)
		self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # 创建路径 '/path/to/data/trainA'
		self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # 创建路径 '/path/to/data/trainB'
		self.dicom = opt.dicom
		self.norm_min = opt.norm_min
		self.norm_max = opt.norm_max

		self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # 从 '/path/to/data/trainA' 加载图片
		self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))	# 从 '/path/to/data/trainB' 加载图片
		self.A_size = len(self.A_paths)  # A 数据集的大小
		self.B_size = len(self.B_paths)  # B 数据集的大小
		btoA = self.opt.direction == 'BtoA'
		input_nc = self.opt.output_nc if btoA else self.opt.input_nc	   # 输入图片的通道数
		output_nc = self.opt.input_nc if btoA else self.opt.output_nc	   # 输出图片的通道数
		self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
		self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

	def __getitem__(self, index):
		""" 返回数据及数据路径

		Parameters:
			index (int)	  -- 用于数据索引的随机整数

		返回包含 A, B, A_paths 和 B_paths 的字典
			A (tensor)	   -- 输入域的一张图片
			B (tensor)	   -- 目标域对应的图片
			A_paths (str)	-- A 的图片路径
			B_paths (str)	-- B 的图片路径
		"""
		A_path = self.A_paths[index % self.A_size]  # 确保 index 索引在数据集大小范围内
		if self.opt.serial_batches:
			index_B = index % self.B_size
		else:   # 从 B 域中随机选取 index 索引,避免固定的成对 A-B 数据
			index_B = random.randint(0, self.B_size - 1)
		B_path = self.B_paths[index_B]

		if not self.dicom:
			A_img = Image.open(A_path).convert('RGB')
			B_img = Image.open(B_path).convert('RGB')
		# TODO: preprocessing read operations
		else:
			A_img = read_dicom(A_path)
			A_img = (A_img - self.norm_min)/(self.norm_max - self.norm_min)*255
			A_img = Image.fromarray(A_img.astype(np.float32), mode='F')
			B_img = read_dicom(B_path)
			B_img = (B_img - self.norm_min)/(self.norm_max - self.norm_min)*255
			B_img = Image.fromarray(B_img.astype(np.float32), mode='F')

		# 对图片应用 transformation（前处理）
		A = self.transform_A(A_img)
		B = self.transform_B(B_img)

		return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

	def __len__(self):
		""" 返回数据集中图片数量（A，B中最大的） """
		return max(self.A_size, self.B_size)
