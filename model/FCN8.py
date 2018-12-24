#coding=utf-8
'''
============================================================================
Fully Convolutional Networks for Semantic Segmentation
============================================================================

'''

import tensorflow as tf
# from model import vgg
from tensorflow.contrib.slim.nets import vgg

slim = tf.contrib.slim

def FCN8(inputs , num_classes):
	'''A TensorFlow implementation of FCN-8s model based on 
		https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn8s/net.py
	
	Args:
		inputs: A 4-D tensor with dimensions [batch_size, height, width, channels]
		num_classes:Integer, the total number of categories in the dataset
	Returns:
		A score map with dimensions [batch_size, height, width, num_classes]

	'''
	inputs = inputs*255
	with slim.arg_scope(vgg.vgg_arg_scope()):
		_, end_points = vgg.vgg_16(inputs,
					               num_classes=1000,
					               is_training=True,
					               dropout_keep_prob=0.5,
					               spatial_squeeze=False,
					               scope='vgg_16')
	pool3 = end_points["vgg_16/pool3"]                      # 1/8H * 1/8 * 256
	pool4 = end_points["vgg_16/pool4"]                      # 1/16H * 1/16W * 512
	pool5 = end_points["vgg_16/pool5"]                      # 1/32H * 1/32W * 512

	with tf.variable_scope('FCN8'): 
		fc6 = slim.conv2d(pool5, 4096, [7,7], scope='fc6')
		dropout6 = slim.dropout(fc6, keep_prob=0.5)
		fc7 = slim.conv2d(dropout6, 4096, [1,1], scope='fc7')
		dropout7 = slim.dropout(fc7, keep_prob =0.5)
		fc8_score = slim.conv2d(dropout7, num_classes, [1,1], activation_fn=None,scope='fc8_score')

		# 2x upsample---->1/16H * 1/16W * num_classes
		upscore2 = slim.conv2d_transpose(fc8_score, num_classes, [4, 4], stride=2, 
										 activation_fn=None, scope="upscore2")
		pool4_score = slim.conv2d(pool4, num_classes, [1,1], activation_fn=None, scope="pool4_score")
		pool4_score_crop = tf.slice(pool4_score, [0, 0, 0, 0], tf.shape(upscore2),name="pool4_score_crop")
		pool4_fuse = tf.add(upscore2, pool4_score_crop, name="pool4_fuse")

		# 2x upsample---->1/8H * 1/8W * num_classes
		upscore4 = slim.conv2d_transpose(pool4_fuse, num_classes, [4, 4], stride=2, 
										 activation_fn=None, scope="upscore4")
		pool3_score = slim.conv2d(pool3, num_classes, [1,1], activation_fn=None, scope="pool3_score")
		pool3_score_crop = tf.slice(pool3_score, [0, 0, 0, 0], tf.shape(upscore4),name="pool3_score_crop")
		pool3_fuse = tf.add(upscore4, pool3_score_crop, name="pool3_fuse")

		# 8x upsample---->H * W * num_classes
		score = slim.conv2d_transpose(pool3_fuse, num_classes, [16, 16], stride=8, 
									  activation_fn=None, scope="score")
		output_shape = tf.stack([tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], num_classes])
		logits = tf.slice(score, [0, 0, 0, 0], output_shape, name = 'logits')
		return logits



if __name__ == '__main__':
	inputs = tf.placeholder(dtype = tf.float32, shape = [4, 224, 224, 3])
	logits = FCN8(inputs = inputs, num_classes = 21)
	print (logits)
