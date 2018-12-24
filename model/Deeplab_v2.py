#coding = utf-8
'''
============================================================================
DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous 
Convolution, and Fully Connected CRFs
============================================================================

'''

import tensorflow as tf

from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.nets import resnet_v2

slim = tf.contrib.slim

def Deeplab_v2(inputs , num_classes, is_training):
	'''A TensorFlow implementation of Deeplab_v2 model based on 
	   http://liangchiehchen.com/projects/DeepLabv2_resnet.html		
	
	Args:
		inputs: A 4-D tensor with dimensions [batch_size, height, width, channels]
		num_classes: Integer, the total number of categories in the dataset
		is_training : Bool, whether to updates the running means and variances during the training.
	Returns:
		A score map with dimensions [batch_size, 1/8*height, 1/8*width, num_classes]

	'''
	with slim.arg_scope(resnet_utils.resnet_arg_scope()) as sc:
		net, _ = resnet_v2.resnet_v2_101(inputs,
					                    num_classes=None,
					                    is_training=is_training,
					                    global_pool=False,
					                    output_stride=8,	
					                    reuse=None,
					                    scope='resnet_v2_101')
	
	# ASPP module without BN layers
	with tf.variable_scope('Deeplab_v2'):
		pool6 = slim.conv2d(net, num_classes, [3,3], rate=6, activation_fn=None, scope='pool6')
		pool12 = slim.conv2d(net, num_classes, [3,3], rate=12, activation_fn=None, scope='pool12')
		pool18 = slim.conv2d(net, num_classes, [3,3], rate=18, activation_fn=None, scope='pool18')
		pool24 = slim.conv2d(net, num_classes, [3,3], rate=24, activation_fn=None, scope='pool24')
		logits = tf.add_n([pool6, pool12, pool18, pool24], name='logits')
		return logits

if __name__ == '__main__':
	inputs = tf.placeholder(dtype = tf.float32, shape = [4, 224, 224, 3])
	logits = Deeplab_v2(inputs, num_classes=21, is_training=True)
	print (logits)