# coding: utf-8
import os
import pathlib
import yaml
import socket
import time
import sys
import logging
import logging.config


# if len(logging.getLogger().handlers) == 0:
# 	logging.basicConfig(level=logging.DEBUG)

class ConfigLoader(object):
	def __init__(self, config_path=None):
		super(ConfigLoader, self).__init__()
		self._config_path = config_path or self._absolute_path('./configs/logger.yaml')
		self._load()
		# self._check_dir()
		try:
			logging.config.dictConfig(self._conf['offline_logging'])
		except Exception as e:
			print("***" * 10)
			print("did not load logger !")
			print("config path: ", self._config_path)
			print("content : ", self._conf['offline_logging'])
			print(e)
			print("***" * 10)
			pass

	def _load(self):
		with open(self._config_path, 'rb') as f:
			self._conf = yaml.safe_load(f)

	@property
	def conf(self):
		return self._conf

config = ConfigLoader('./configs/logger.yaml')