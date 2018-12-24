import tensorflow as tf
import numpy as np


def test():
	inputs = tf.constant([4,5,2,3,0,1,6,7,12,13,10,11,8,9,14,15,16,17],dtype = tf.float32,shape = [2,3,3,1])
	
	one_like_mask = tf.ones_like(inputs, dtype=tf.int32)
	input_shape = tf.shape(inputs)
	output_shape = (input_shape[0], input_shape[1] * 2, input_shape[2] * 2, input_shape[3])
	batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], 0)
	batch_range = tf.reshape(tf.range(input_shape[0], dtype=tf.int32), shape=batch_shape)
	b = one_like_mask * batch_range
	y = mask // (output_shape[2] * output_shape[3])
	x = (mask // output_shape[3]) % output_shape[2] #mask % (output_shape[2] * output_shape[3]) // output_shape[3]
	feature_range = tf.range(output_shape[3], dtype=tf.int32)
	f = one_like_mask * feature_range
	with tf.Session() as sess:
		print(batch_shape.eval())
		print ('=======')
		print (batch_range.eval().shape)
		print ('=======')
		print (b.eval())
		print ('=======')
		print (y.eval())
		print ('=======')
		print (x.eval())
		print ('=======')
		print (f.eval())


if __name__ == '__main__':
	inputs = tf.constant([4,5,2,3],dtype = tf.float32,shape = [2,2])
	x = inputs//2
	with tf.Session() as sess:
		print (x.eval())