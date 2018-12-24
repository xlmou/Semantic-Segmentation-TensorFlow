#coding=utf-8
'''
================================================================================
SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
================================================================================

A TensorFlow implementation of the standard version of Seg_Net model. There are three Seg_Net model :
	1)basic version: https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Models/segnet_basic_train.prototxt
	2)standard version: https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Models/segnet_train.prototxt
	3)bayesian version: https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Models/bayesian_segnet_train.prototxt
'''

import tensorflow as tf
slim = tf.contrib.slim

def Seg_Net_arg_scope(weight_decay=0.0001,
		             batch_norm_decay=0.997,
		             batch_norm_epsilon=1e-5,
		             batch_norm_scale=True,
		             activation_fn=tf.nn.relu,
		             use_batch_norm=True,
		             batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):
	'''Defines the default Seg_Net arg scope.
	'''
	batch_norm_params = {
		'decay': batch_norm_decay,
		'epsilon':batch_norm_epsilon,
		'scale':batch_norm_scale,
		'updates_collections': batch_norm_updates_collections,
		'fused': None,  # Use fused batch norm if possible.
	}
	with slim.arg_scope([slim.conv2d],
		weights_regularizer=slim.l2_regularizer(weight_decay),
		weights_initializer=slim.variance_scaling_initializer(),
		activation_fn=activation_fn,
		normalizer_fn=slim.batch_norm if use_batch_norm else None,
		normalizer_params=batch_norm_params):
		with slim.arg_scope([slim.batch_norm], **batch_norm_params):
			with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
				return arg_sc



def unpool(updates, mask, k_size=[1, 2, 2, 1], output_shape=None, scope=''):
    '''Unpooling function based on the implementation by Panaetius at 
       https://github.com/tensorflow/tensorflow/issues/2169
       https://github.com/yselivonchyk/Tensorflow_WhatWhereAutoencoder/blob/master/WhatWhereAutoencoder.py

    Args:
    - inputs: a 4D tensor of shape [batch_size, height, width, num_channels] 
    - mask: a 4D tensor that represents the argmax values/pooling indices of 
           the previously max-pooled layer, which has the same shape with inputs
    - k_size: a list of values representing the dimensions of the unpooling filter.
    - output_shape: a list of values to indicate what the final output shape should be after unpooling
    - scope : the string name to name your scope

    Returns:
    - A 4D tensor that has the shape of output_shape.

    '''
    with tf.variable_scope(scope):
        mask = tf.cast(mask, tf.int32)
        input_shape = tf.shape(updates, out_type=tf.int32)    

        # calculation indices for batch, height, width and channels
        one_like_mask = tf.ones_like(mask, dtype=tf.int32)
        batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], 0)
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int32), shape=batch_shape)
        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = (mask // output_shape[3]) % output_shape[2] 
        feature_range = tf.range(output_shape[3], dtype=tf.int32)
        f = one_like_mask * feature_range

        # stack the indices of batch, height, width and channels & reshape & transpose
        updates_size = tf.size(updates)
        indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
        values = tf.reshape(updates, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        return ret


def Seg_Net(inputs , num_classes):
	'''	
	Args:
		inputs: A 4-D tensor with dimensions [batch_size, height, width, channels]
		num_classes:Integer, the total number of categories in the dataset
	Returns:
		A score map with dimensions [batch_size, height, width, num_classes]

	'''
	with slim.arg_scope(Seg_Net_arg_scope()):
		with tf.variable_scope("Seg_Net") as sc:
			end_points_collection = sc.original_name_scope + '_end_points'
			with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.conv2d_transpose],
								outputs_collections=end_points_collection):

				# encoder is similar to VGG16 except for the added BN layer.
				net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
				shape1 = tf.shape(net)
				net, pool1_indice = tf.nn.max_pool_with_argmax(net,
                                                               ksize=[1,2,2,1],
                                                               strides=[1,2,2,1],
                                                               padding='SAME',
                                                               name='pool1')
				net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
				shape2 = tf.shape(net)
				net, pool2_indice = tf.nn.max_pool_with_argmax(net,
                                                               ksize=[1,2,2,1],
                                                               strides=[1,2,2,1],
                                                               padding='SAME',
                                                               name='pool2')
				net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
				shape3 = tf.shape(net)
				net, pool3_indice = tf.nn.max_pool_with_argmax(net,
                                                               ksize=[1,2,2,1],
                                                               strides=[1,2,2,1],
                                                               padding='SAME',
                                                               name='pool3')
				net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
				shape4 = tf.shape(net)
				net, pool4_indice = tf.nn.max_pool_with_argmax(net,
                                                               ksize=[1,2,2,1],
                                                               strides=[1,2,2,1],
                                                               padding='SAME',
                                                               name='pool4')
				net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
				net = slim.max_pool2d(net, [2, 2], scope='pool5')

				# decoder is symmetrical with encoder.
				net = slim.conv2d_transpose(net, 512, [4, 4], stride=2, 
											activation_fn=None, scope="upsample5")
				net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5_d')
				net = unpool(net, pool4_indice, output_shape=shape4, scope = 'upsample4')

				net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4_1_d')
				net = slim.conv2d(net, 256, [3,3], scope='conv4_2_d')
				net = unpool(net, pool3_indice, output_shape=shape3, scope = 'upsample3')

				net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3_1_d')
				net = slim.conv2d(net, 128, [3,3], scope='conv3_2_d')
				net = unpool(net, pool2_indice, output_shape=shape2, scope = 'upsample2')

				net = slim.conv2d(net, 128, [3, 3], scope='conv2_1_d')
				net = slim.conv2d(net, 64, [3, 3], scope='conv2_2_d')
				net = unpool(net, pool1_indice,output_shape=shape1, scope = 'upsample1')

				net = slim.conv2d(net, 64, [3, 3], scope='conv1_d')

				end_points = slim.utils.convert_collection_to_dict(end_points_collection)

				logits = slim.conv2d(net, num_classes, [3,3], normalizer_fn=None, 
									 activation_fn=None, scope='logits')
				return logits



if __name__ == '__main__':
	inputs = tf.placeholder(dtype = tf.float32, shape = [4, 224, 224, 3])
	logits = Seg_Net(inputs = inputs, num_classes = 21)
	print (logits)