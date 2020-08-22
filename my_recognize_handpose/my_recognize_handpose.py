import cv2
import numpy as np
import keras
import tensorflow as tf

def classify_test(model, img):
	img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	img = cv2.flip(img, 1)

	# Reshape
	res = cv2.resize(img, (28,28), interpolation=cv2.INTER_AREA)

	# Convert to float values between 0 and 1
	res = res.astype(dtype='float64')
	res = res / 255
	res = np.reshape(res, (1,28,28,1))

	prediction = model.predict(res)	# (1, 9)

	sort_prediction_index = np.argsort(prediction[0])

	return sort_prediction_index[-1]

if __name__ == '__main__':
	print('>> Loading keras model for pose classification')
	try:
		model = keras.models.load_model('model/my_hand_poses_recognition_10.h5')
	except Exception as e:
		print('model loading is failed\n')
		print(e)
	
	index = {0:'Cursor', 1:'Enter', 2:'Esc', 3:'LeftClick', 4:'WheelClick', 5:'RightClick', 6:'ScrollUp', 7:'ScrollDown', 8:'Spacebar'}
	img = cv2.imread('images/spacebar_0.png')

	print(index[classify_test(model, img)])
	cv2.imshow('test', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()