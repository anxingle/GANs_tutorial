from __future__ import print_function
import os
import tarfile
import requests
from warnings import warn
from zipfile import ZipFile
from bs4 import BeautifulSoup
from os.path import abspath, isdir, join, basename


class GetData(object):
	""" 下载 CycleGAN or pix2pix 数据集.

	Parameters:
		technique (str) -- 'cyclegan' or 'pix2pix' 中一个.
		verbose (bool)  -- 打印额外信息.

	Examples:
		>>> from util.get_data import GetData
		>>> gd = GetData(technique='cyclegan')
		>>> new_data_path = gd.get(save_path='./datasets')  # 会显示选项信息
	"""

	def __init__(self, technique='cyclegan', verbose=True):
		url_dict = {
			'pix2pix': 'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/',
			'cyclegan': 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets'
		}
		self.url = url_dict.get(technique.lower())
		self._verbose = verbose

	def _print(self, text):
		if self._verbose:
			print(text)

	@staticmethod
	def _get_options(r):
		soup = BeautifulSoup(r.text, 'lxml')
		options = [h.text for h in soup.find_all('a', href=True)
				   if h.text.endswith(('.zip', 'tar.gz'))]
		return options

	def _present_options(self):
		r = requests.get(self.url)
		options = self._get_options(r)
		print('Options:\n')
		for i, o in enumerate(options):
			print("{0}: {1}".format(i, o))
		choice = input("\nPlease enter the number of the "
					   "dataset above you wish to download:")
		return options[int(choice)]

	def _download_data(self, dataset_url, save_path):
		if not isdir(save_path):
			os.makedirs(save_path)

		base = basename(dataset_url)
		temp_save_path = join(save_path, base)

		with open(temp_save_path, "wb") as f:
			r = requests.get(dataset_url)
			f.write(r.content)

		if base.endswith('.tar.gz'):
			obj = tarfile.open(temp_save_path)
		elif base.endswith('.zip'):
			obj = ZipFile(temp_save_path, 'r')
		else:
			raise ValueError("Unknown File Type: {0}.".format(base))

		self._print("Unpacking Data...")
		obj.extractall(save_path)
		obj.close()
		os.remove(temp_save_path)

	def get(self, save_path, dataset=None):
		"""
		下载数据集.

		Parameters:
			save_path (str) -- 将要保存数据的目录.
			dataset (str)   -- (可选). 指定要下载的数据集.
							注意: 必须包含文件扩展名。否则将会展示需要你选择的选项

		Returns:
			save_path_full (str) -- 下载好的数据的绝对路径
		"""
		if dataset is None:
			selected_dataset = self._present_options()
		else:
			selected_dataset = dataset

		save_path_full = join(save_path, selected_dataset.split('.')[0])

		if isdir(save_path_full):
			warn("\n'{0}' already exists. Voiding Download.".format(
				save_path_full))
		else:
			self._print('Downloading Data...')
			url = "{0}/{1}".format(self.url, selected_dataset)
			self._download_data(url, save_path=save_path)

		return abspath(save_path_full)
