#coding = utf-8
'''
============================================================================
Pyramid Scene Parsing Network
============================================================================

'''

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.nets import resnet_v2

slim = tf.contrib.slim

def InterpBlock(net, k_h, k_w, s_h, s_w, shape, name, padding='SAME'):
	'''
	   a branch of pyramid pooling module
	''' 
	with tf.variable_scope(name) as sc:
		net = slim.avg_pool2d(net, [k_h,k_w], stride=[s_h,s_w], padding=padding)
		net = slim.conv2d(net, 512, [1, 1], activation_fn=None)
		net = slim.batch_norm(net, fused=True)
		net = tf.nn.relu(net)
		net = tf.image.resize_bilinear(net, size=shape)
		return net

def PyramidPoolingModule(inputs, shape):
	"""
	  build pyramid pooling module
	"""
	pool60 = InterpBlock(inputs, 60, 60, 60, 60, shape, name='pool60')
	pool30 = InterpBlock(inputs, 30, 30, 30, 30, shape, name='pool30')
	pool20 = InterpBlock(inputs, 20, 20, 20, 20, shape, name='pool20')
	pool10 = InterpBlock(inputs, 10, 10, 10, 10, shape, name='pool10')

	concat = tf.concat([inputs, pool60, pool30, pool20, pool10], axis=-1, name='concat')
	return concat

def PSPNet(inputs , num_classes, is_training):
	'''A TensorFlow implementation of PSPNet model based on 
	   https://github.com/hszhao/PSPNet/tree/master/evaluation/prototxt	   	
	
	Args:
		inputs: A 4-D tensor with dimensions [batch_size, height, width, channels]
		num_classes: Integer, the total number of categories in the dataset
		is_training : Bool, whether to updates the running means and variances during the training.
	Returns:
		A score map with dimensions [batch_size, 1/8*height, 1/8*width, num_classes]

	'''
	with slim.arg_scope(resnet_utils.resnet_arg_scope()):
		net, end_points = resnet_v2.resnet_v2_101(inputs,
							                      num_classes=None,
							                      is_training=is_training,
							                      global_pool=False,
							                      output_stride=8,
							                      reuse=None,
							                      scope='resnet_v2_101')
	with tf.variable_scope("PSPNet") as sc:    	
		shape = tf.shape(net)[1:3]
		net = PyramidPoolingModule(net, shape = shape)
		net = slim.conv2d(net, 512, [3, 3], activation_fn=None, scope='conv5')
		net = slim.batch_norm(net, fused=True, scope='conv5_bn')
		net = tf.nn.relu(net, name='conv5_bn_relu')
		logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
		return logits


if __name__ == '__main__':
	inputs = tf.placeholder(dtype = tf.float32, shape = [4, 224, 224, 3])
	logits = PSPNet(inputs, num_classes=21, is_training=True)
	print (logits)