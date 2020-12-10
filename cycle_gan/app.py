"""
CycleGAN 服务后端脚本
"""
import os
import logging
from util.load_conf import config
import cv2
import json

_logger = logging.getLogger(__name__)
_logger.error("*****ok*********")

from werkzeug.utils import secure_filename
from flask import Flask, request, Response, send_from_directory
from flask_cors import CORS
import time
import random

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import htmls
from util import util


opt = TestOptions().parse()
# 直接指定一些选项进行测试
opt.num_threads = 0   # 测试时 DataLoader 仅支持 num_threads = 0
opt.batch_size = 1	# 测试时batch_size设为1
opt.serial_batches = True  # 关闭数据 shuffling; 需要随机选择图片时再打开
opt.no_flip = True	# no flip; 如果需要翻转图片，注释本行
opt.display_id = -1   # no visdom display; 测试时将结果保存入 HTML 文件

model = create_model(opt)	  # 根据 opt.model 和其他选项创建 model 实例
model.setup(opt)			   # 常规设置: 加载和打印网络; 创建 schedulers
model.eval()

static_folder = 'static'
app = Flask(__name__, static_folder=static_folder)
CORS(app)


@app.route('/', methods=['GET'])
def index():
	return app.send_static_file('index.html')


@app.route('/p', methods=['POST'])
def predict():
	_logger.info('start predict...')
	res = {}
	errors = []

	if request.method == 'POST':
		if 'image' not in request.files:
			errors.append('要求有image字段，类型为multipart/form-data, 对应上传文件')
		else:
			f = request.files['image']
			file_path = './static/' + secure_filename(f.filename)
			_logger.info('file_path: %s' % file_path)
			f.save(file_path)
			# img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
			img = util.im2tensor(file_path)
			input_data = {'A': img, 'A_paths': file_path}
			model.set_input(input_data)
			model.test()
			visuals = model.get_current_visuals() # 得到结果图片
			result_img = util.tensor2im(visuals['fake'])
			save_path = time.strftime('%H_%M_%S', time.localtime()) + '_%d.bmp' % random.randint(0, 12)
			if result_img is not None:
				cv2.imwrite('./static/%s' % save_path, result_img)
				result = '/static/%s' % save_path
	if len(errors) > 0:
		res['msg'] = ';'.join(errors)
		res['status'] = False
	else:
		res['status'] = True
		res['img'] = result
	return Response(response=json.dumps(res), status=200, mimetype='application/json')


if __name__ == '__main__':
	# [pix2pix]: 在原始 pix2pix 中使用 batchnorm 和 dropout。可以试试使用和不使用 eval() 模式的区别。
	# [CycleGAN]: 不会影响CycleGAN。因为CycleGAN 使用 instancenorm，没有使用 dropout。
	app.run(host='0.0.0.0', port='8080', debug=True)
