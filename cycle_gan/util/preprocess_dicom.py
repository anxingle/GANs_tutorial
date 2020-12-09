from pathlib import Path
import numpy as np
import SimpleITK as sitk
import os
import h5py
import cv2
import pydicom
import time


def read_dicom(path, image_type='ct', _type=np.float32):
	assert 'ct' in image_type.lower() or 'mr' in image_type.lower(), 'read_dicom only read CT/MR!'
	start = time.time()
	ds = sitk.ReadImage(path)
	print('read cost time: %.2f ' % (time.time() - start))
	# ds = pydicom.read_file(path, force=True)
	# print('pixelData' in ds)
	# ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

	try:
		if str(image_type).lower() == 'ct':
			pixel_array = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
		else:
			pixel_array = sitk.GetArrayViewFromImage(ds)
			print('Get Array cost time: %.2f ' % (time.time() - start))
			# pixel_array = ds.pixel_array
			# pixel_array = ds.PixelData
	except Exception as e:
		return None

	pixel_array = np.array(pixel_array[0], dtype=_type)
	# pixel_array2 = [y for y in pixel_array]
	return pixel_array


def convertA(path):
	path = Path(path)
	files = [f for f in path.iterdir() if f.is_file() and '.DS' not in str(f)]
	error = 0

	for n, file in enumerate(files):
		raw_data = read_dicom(str(file), image_type='mr')
		if raw_data is None:
			error += 1
			print('raw_data is None!')
			continue

		min_value, max_value = raw_data.min(), raw_data.max()
		im = ((raw_data-min_value)*1.0/(max_value - min_value)*255.0)
		# im = np.rot90(im, 2)
		img_resize = cv2.resize(im, (256, 256))
		img_resize = img_resize.astype(np.uint8)
		cv2.imwrite('./dataA/a11_%d.bmp' % n, img_resize)
	print('error:', error, ' all: ', n)


def convertB(path):
	path = Path(path)
	dirs = [f for f in path.iterdir() if f.is_dir()]
	for n, sub_dir in enumerate(dirs):
		file = sub_dir/'ct_xray_data.h5'
		with h5py.File(str(file), 'r') as hdf5:
			x_ray1 = np.asarray(hdf5['xray1'])
			x_ray2 = np.asarray(hdf5['xray2'])
			cv2.imwrite('./data/xray1_%d.bmp' % n, x_ray1)
			cv2.imwrite('./data/xray2_%d.bmp' % n, x_ray2)
			print(n)


if __name__ == '__main__':
	# convertB('/Volumes/data/workspace/gan/X2CT_data/LIDC-HDF5-256/data/LIDC-HDF5-256/')
	convertA('/Volumes/data/CT_data/301-lung-xrays/zheng')