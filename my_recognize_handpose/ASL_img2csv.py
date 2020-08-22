import numpy as np
import sys
import os
import csv
import cv2

folder_index = {'Cursor':0, 'Enter':1, 'Esc':2, 'LeftClick':3, 'WheelClick':4, 'RightClick':5, 'ScrollUp':6, 'ScrollDown':7, 'Spacebar':8}

def createFileList(my_dir, format='.png'):
	fileList = []
	print(my_dir)

	for root, dirs, files in os.walk(my_dir, topdown=False):
		for name in files:
			if name.endswith(format):
				fullname = os.path.join(root, name)
				fileList.append(fullname)

	return fileList

if __name__ == '__main__':
	dir_url = 'Real_HandPose_dataset/test/'

	folder_list = []

	for folders in os.listdir(dir_url):
		folder_list.append(folders)
	print(folder_list)

	for folder in folder_list:
		my_fileList = createFileList(dir_url + folder + '/')
		print(my_fileList)

		for file in my_fileList:
			print(file)
			img = cv2.imread(file)
			#cv2.imshow('img', img)
			#cv2.waitKey(0)

			img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
			img = cv2.flip(img, 1)
			img = cv2.resize(img, (28,28), interpolation=cv2.INTER_AREA)

			res = img.astype(dtype='int32')
			res = np.reshape(res, (1,28,28,1))

			res = res.flatten()
			#print(res)

			command_label = np.array([folder_index[folder]])
			#print(command_label)
			res = np.append(command_label, res)
			#print(res)

			with open('Real_HandPose_dataset/real_dataset_test.csv', 'a') as f:
				writer = csv.writer(f)
				writer.writerow(res)