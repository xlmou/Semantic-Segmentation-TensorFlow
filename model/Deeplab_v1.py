#coding = utf-8
'''
============================================================================
Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs
============================================================================

'''

import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg

slim = tf.contrib.slim

def Deeplab_v1(inputs , num_classes):
	'''A TensorFlow implementation of Deeplab_v1 model based on
	   http://liangchiehchen.com/projects/DeepLab-LargeFOV.html 
	
	Args:
		inputs: A 4-D tensor with dimensions [batch_size, height, width, channels]
		num_classes: Integer, the total number of categories in the dataset
	Returns:
		A score map with dimensions [batch_size, 1/8*height, 1/8*width, num_classes]

	'''
	with slim.arg_scope(vgg.vgg_arg_scope()):
		_, end_points = vgg.vgg_16(inputs,
					               num_classes=1000,
					               is_training=True,
					               dropout_keep_prob=0.5,
					               spatial_squeeze=False,
					               scope='vgg_16')
	conv4 = end_points["vgg_16/conv4/conv4_3"]                   # 1/8H * 1/8 * 256
	
	with tf.variable_scope('Deeplab_v1'):
		pool4 = slim.max_pool2d(conv4, [3,3], stride=1, padding='SAME', scope='pool4')

		conv5_1 = slim.conv2d(pool4, 512, [3,3], rate=2, scope='conv5_1')
		conv5_2 = slim.conv2d(conv5_1, 512, [3,3], rate=2, scope='conv5_2')
		conv5_3 = slim.conv2d(conv5_2, 512, [3,3], rate=2, scope='conv5_3')
		pool5_1 = slim.max_pool2d(conv5_3, [3,3], stride=1, padding='SAME', scope='pool5_1')
		pool5_2 = slim.avg_pool2d(pool5_1, [3,3], stride=1, padding='SAME', scope='pool5_2')

		conv6 = slim.conv2d(pool5_2, 1024, [3,3], rate=12, scope='conv6')
		dropout1 = slim.dropout(conv6, keep_prob=0.5)
		conv7 = slim.conv2d(dropout1, 1024, [1,1], scope='conv7')
		dropout2 = slim.dropout(conv7, keep_prob=0.5)
		logits = slim.conv2d(dropout2, num_classes, [1,1],activation_fn=None, scope='logits')
		return logits


if __name__ == '__main__':
	inputs = tf.placeholder(dtype = tf.float32, shape = [4, 224, 224, 3])
	logits = Deeplab_v1(inputs, num_classes=21)
	print (logits)

