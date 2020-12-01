"""
本训练脚本可训练多种模型 (option '--model': pix2pix, cyclegan) 以及多种不同的数据集 (option 'dataset_mode': aligned, unaligned, single)。

首先根据命令行参数创建 模型，数据集实例 和 可视化实例。
支持标准化训练流程，并可在训练过程中可视化/保存图片，打印/保存 loss曲线，保存模型。
支持 断点训练。可使用 '--continue_train' 来恢复之前的训练。

示例:
	Train a CycleGAN model:
		python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
	Train a pix2pix model:
		python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

参阅文件 options/base_options.py & options/train_options.py 了解更多训练参数选项说明.
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
	opt = TrainOptions().parse()
	dataset = create_dataset(opt)  # 根据 opt.dataset_mode 和其他参数创建数据集
	dataset_size = len(dataset)	# 数据集中图片数
	print('The number of training images = %d' % dataset_size)

	model = create_model(opt)		# 根据 opt.model 及其他参数创建模型类
	model.setup(opt)				# 常规设置：加载、打印网络；创建 schedulers
	visualizer = Visualizer(opt)	# 创建 visualizer 以显示/保存 图片及曲线
	total_iters = 0					# 训练迭代的次数

	for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):	# 最外层循环 通过<epoch_count> , <epoch_count>+<save_latest_freq> 次保存模型
		epoch_start_time = time.time()	# 每次epoch开始的计时器
		iter_data_time = time.time()	# 每次迭代数据加载的计时器
		epoch_iter = 0					# 当前epoch的迭代次数，每次epoch重置为0
		visualizer.reset()				# 重置 visualizer: 确保将结果保存到 HTML
		model.update_learning_rate()	# 每个 epoch 开始更新 learning rates
		for i, data in enumerate(dataset):  # 每个 epoch 内开始迭代
			iter_start_time = time.time()  # 每个迭代的计算耗时
			if total_iters % opt.print_freq == 0:
				t_data = iter_start_time - iter_data_time

			total_iters += opt.batch_size
			epoch_iter += opt.batch_size
			model.set_input(data)		 # 从 dataset 实例解数据并预处理
			model.optimize_parameters()		# 计算损失，得到梯度，更新网络权重

			if total_iters % opt.display_freq == 0:		# 在visdom上显示图片，保存图片到 HTML 文件中
				save_result = total_iters % opt.update_html_freq == 0
				model.compute_visuals()
				visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

			if total_iters % opt.print_freq == 0:	# 打印训练损失并保存 logging 信息到硬盘
				losses = model.get_current_losses()
				t_comp = (time.time() - iter_start_time) / opt.batch_size
				visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
				if opt.display_id > 0:
					visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

			if total_iters % opt.save_latest_freq == 0:   # 每 <save_latest_freq> 次迭代保存最新模型
				print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
				save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
				model.save_networks(save_suffix)

			iter_data_time = time.time() # 每次迭代数据加载的计时器
		if epoch % opt.save_epoch_freq == 0:		# 每 <save_epoch_freq> 次 epoch 保存最新模型
			print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
			model.save_networks('latest')
			model.save_networks(epoch)

		print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
