import numpy as np
import sys
import os

if __name__ == '__main__':
	dir_url = 'Real_HandPose_dataset/train/'

	for folder in os.listdir(dir_url):
		for file in os.listdir(dir_url + folder + '/'):
			os.rename(dir_url + folder + '/' + file, dir_url + folder + '/' + file.replace('png', 'jpg'))