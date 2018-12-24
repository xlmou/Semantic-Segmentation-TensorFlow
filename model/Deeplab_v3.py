#coding = utf-8
'''
============================================================================
Rethinking Atrous Convolution for Semantic Image Segmentation
============================================================================
  
  Deeplab v3 has proposed two models based on deeplab v2. The first one is a cascaded model 
and the second one is an augmented ASPP model which is slight better than the first 
in terms of accuracy, therefore this is a tensorflow implementation of second model.
'''
import tensorflow as tf
from model import resnet_utils
from model import resnet_v2

slim = tf.contrib.slim


def atrous_spatial_pyramid_pooling(net, depth, scope):
    """ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with 
    rates = (6, 12, 18) when output stride = 16 (all with 256 filters and batch normalization),
    and (b) the image-level features as described in https://arxiv.org/abs/1706.05587
    
    Args:
	    net: tensor of shape [BATCH_SIZE, WIDTH, HEIGHT, DEPTH]
	    scope: scope name of the aspp layer
	Return: 
		network layer with aspp applyed to it.
    """

    with tf.variable_scope(scope):        

        # apply global average pooling
        shape = tf.shape(net)[1:3]
        global_pool = tf.reduce_mean(net, [1, 2], keepdims=True, name='global_pool')
        global_pool = slim.conv2d(global_pool, depth, [1,1], 
        						  activation_fn=None, scope="global_pool_conv1x1")
        global_pool = tf.image.resize_bilinear(global_pool, size=shape)

        pool1 = slim.conv2d(net, depth, [1,1], activation_fn=None, scope="pool1")
        pool6 = slim.conv2d(net, depth, [3,3], rate=6, activation_fn=None, scope="pool6")
        pool12 = slim.conv2d(net, depth, [3,3], rate=12, activation_fn=None, scope="pool12")
        pool18 = slim.conv2d(net, depth, [3,3], rate=18, activation_fn=None, scope="pool18")

        net = tf.concat([global_pool, pool1, pool6, pool12, pool18], axis=3, name="concat")
        net = slim.conv2d(net, depth, [1, 1], activation_fn=None, scope="output_conv_1x1")
        return net


def Deeplab_v3(inputs , num_classes, is_training):
	'''A TensorFlow implementation of Deeplab_v3 model based on
	   https://github.com/sthalles/deeplab_v3 	   	
	
	Args:
		inputs: A 4-D tensor with dimensions [batch_size, height, width, channels]
		num_classes: Integer, the total number of categories in the dataset
		is_training : Bool, whether to updates the running means and variances during the training.
	Returns:
		A score map with dimensions [batch_size, 1/8*height, 1/8*width, num_classes]

	'''
	with slim.arg_scope(resnet_utils.resnet_arg_scope()) as sc:
		net, end_points = resnet_v2.resnet_v2_101(inputs,
							                      num_classes=None,
							                      is_training=is_training,
							                      multi_grid=[1, 2, 4],
							                      global_pool=False,
							                      output_stride=16,
							                      reuse=None,
							                      scope='resnet_v2_101')
		block4 = end_points['resnet_v2_101/block4']       # (14, 14, 2048)
		
		with tf.variable_scope('Deeplab_v3'):
			net = atrous_spatial_pyramid_pooling(block4, depth=256, scope="ASPP")
			logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
			return logits


if __name__ == '__main__':
	inputs = tf.placeholder(dtype = tf.float32, shape = [4, 224, 224, 3])
	logits = Deeplab_v3(inputs, num_classes=21, is_training=True)
	print (logits)