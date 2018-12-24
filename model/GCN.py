#coding = utf-8
'''
============================================================================
Large Kernel Matters——
Improve Semantic Segmentation by Global Convolutional Network
============================================================================

'''

import tensorflow as tf

from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.nets import resnet_v2

slim = tf.contrib.slim

def BoundaryRefinementBlock(inputs, num_classes, kernel_size=[3, 3]):
	"""
	Boundary Refinement Block for GCN. 
	"""
	net = slim.conv2d(inputs, num_classes, kernel_size)
	net = slim.conv2d(net, num_classes, kernel_size, activation_fn=None)
	net = tf.add(inputs, net)
	return net


def GlobalConvBlock(inputs, num_classes=21, kernel_size=15):
	"""Global Convolution Block for GCN
	
	Args:
		inputs: A 4-D tensor with dimention [batch_size, height, width, channels]
		num_classes: Integer, the total number of categories in the dataset
		kernel_size: Integer, the kernel_size of Global Convolution Block 
	"""
	net_1 = slim.conv2d(inputs, num_classes, [kernel_size, 1], activation_fn=None)
	net_1 = slim.conv2d(net_1, num_classes, [1, kernel_size], activation_fn=None)
	
	net_2 = slim.conv2d(inputs, num_classes, [1, kernel_size], activation_fn=None)
	net_2 = slim.conv2d(net_2, num_classes, [kernel_size, 1], activation_fn=None)
	
	net = tf.add(net_1, net_2)
	return net


def GCN(inputs, num_classes, is_training):
	'''A TensorFlow implementation of GCN model based on 
	   "Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network"
	
	Args:
		inputs: A 4-D tensor with dimensions [batch_size, height, width, channels]
		num_classes: Integer, the total number of categories in the dataset
		is_training : Bool, whether to updates the running means and variances during the training.
	Returns:
		A score map with dimensions [batch_size, height, width, num_classes]

	'''
	with slim.arg_scope(resnet_utils.resnet_arg_scope()):
		net, end_points = resnet_v2.resnet_v2_101(inputs,
							                      num_classes=None,
							                      is_training=is_training,
							                      global_pool=False,
							                      output_stride=None,
							                      reuse=None,
							                      scope='resnet_v2_101')
	block1 = end_points["resnet_v2_101/block1/unit_2/bottleneck_v2"]    # (56, 56, 256)
	block2 = end_points["resnet_v2_101/block2/unit_3/bottleneck_v2"]    # (28, 28, 512)
	block3 = end_points["resnet_v2_101/block3/unit_22/bottleneck_v2"]   # (14, 14, 1024)
	block4 = net                                                        # (7, 7, 2048)

	with tf.variable_scope("gcn") as sc:
		down5 = GlobalConvBlock(block4, num_classes=num_classes)
		down5 = BoundaryRefinementBlock(down5, num_classes=num_classes, kernel_size=[3,3])
		down5 = slim.conv2d_transpose(down5, num_classes, kernel_size=[4, 4], 
									  stride=2, activation_fn=None)      # (14, 14, 21)

		down4 = GlobalConvBlock(block3, num_classes=num_classes)
		down4 = BoundaryRefinementBlock(down4, num_classes=num_classes, kernel_size=[3,3])
		down4 = tf.add(down4, down5)
		down4 = BoundaryRefinementBlock(down4, num_classes=num_classes, kernel_size=[3,3])
		down4 = slim.conv2d_transpose(down4, num_classes, kernel_size=[4, 4], 
									  stride=2, activation_fn=None)     # (28, 28, 21)

		down3 = GlobalConvBlock(block2, num_classes=num_classes)
		down3 = BoundaryRefinementBlock(down3, num_classes=num_classes, kernel_size=[3,3])
		down3 = tf.add(down3, down4)
		down3 = BoundaryRefinementBlock(down3, num_classes=num_classes, kernel_size=[3,3])
		down3 = slim.conv2d_transpose(down3, num_classes, kernel_size=[4, 4], 
									  stride=2, activation_fn=None)     # (56, 56, 21)

		down2 = GlobalConvBlock(block1, num_classes=num_classes)
		down2 = BoundaryRefinementBlock(down2, num_classes=num_classes, kernel_size=[3,3])
		down2 = tf.add(down2, down3)
		down2 = BoundaryRefinementBlock(down2, num_classes=num_classes, kernel_size=[3,3])
		down2 = slim.conv2d_transpose(down2, num_classes, kernel_size=[4, 4], 
									  stride=2, activation_fn=None)     # (112, 112, 21)

		output = BoundaryRefinementBlock(down2, num_classes=num_classes, kernel_size=[3,3])
		output = slim.conv2d_transpose(output, num_classes, kernel_size=[4, 4], 
									   stride=2, activation_fn=None)    # (224, 224, 21)
		output = BoundaryRefinementBlock(output, num_classes=num_classes, kernel_size=[3,3])
		return output



if __name__ == '__main__':
	inputs = tf.placeholder(dtype = tf.float32, shape = [4, 224, 224, 3])
	logits = GCN(inputs, num_classes=21, is_training=True)
	print (logits)