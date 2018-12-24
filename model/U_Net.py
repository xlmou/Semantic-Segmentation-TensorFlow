#coding=utf-8
'''
============================================================================
U-Net: Convolutional Networks for Biomedical Image Segmentation
============================================================================

'''

import tensorflow as tf
slim = tf.contrib.slim


def U_Net(inputs , num_classes):
	'''A TensorFlow implementation of U_Net model based on 
		https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
	
	Args:
		inputs: A 4-D tensor with dimensions [batch_size, height, width, channels]
		num_classes:Integer, the total number of categories in the dataset
	Returns:
		A score map with dimensions [batch_size, height, width, num_classes]

	'''
	inputs = inputs*255
	with tf.variable_scope("U_Net") as sc:
		end_points_collection = sc.original_name_scope + '_end_points'
		with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.conv2d_transpose, slim.dropout],
							outputs_collections=end_points_collection):

			# contracting path
			conv1 = slim.repeat(inputs, 2, slim.conv2d, 64, [3,3], scope='conv1') # H * W * 64
			pool1 = slim.max_pool2d(conv1, [2,2], scope = 'pool1')

			conv2 = slim.repeat(pool1, 2, slim.conv2d, 128, [3,3], scope='conv2') # 1/2H * 1/2W * 128
			pool2 = slim.max_pool2d(conv2, [2,2], scope = 'pool2')

			conv3 = slim.repeat(pool2, 2, slim.conv2d, 256, [3,3], scope='conv3') # 1/4H * 1/4W * 256
			pool3 = slim.max_pool2d(conv3, [2,2], scope = 'pool3')

			conv4 = slim.repeat(pool3, 2, slim.conv2d, 512, [3,3], scope='conv4') 
			dropout4 = slim.dropout(conv4, keep_prob = 0.5)                        # 1/8H * 1/8W * 512
			pool4 = slim.max_pool2d(dropout4, [2,2], scope ='pool4')	

			conv5 = slim.repeat(pool4, 2, slim.conv2d, 1024, [3,3], scope='conv5') # 1/16H * 1/16W * 1024
			dropout5 = slim.dropout(conv5, keep_prob = 0.5)

			# expanding path
			upsample1 = slim.conv2d_transpose(dropout5, 512, [4, 4], stride=2, scope="upsample1" )
			crop1 = tf.slice(upsample1, [0, 0, 0, 0], tf.shape(dropout4), name='crop1')
			concat1 = tf.concat([dropout4, crop1], axis = 3)             # 1/8H * 1/8W * 1024                   
			conv4_d = slim.repeat(concat1, 2, slim.conv2d, 512, [3,3], scope='conv4_d') 

			upsample2 = slim.conv2d_transpose(conv4_d, 256, [4,4], stride=2, scope='upsample2')
			crop2 = tf.slice(upsample2, [0, 0, 0, 0], tf.shape(conv3), name='crop2')
			concat2 = tf.concat([conv3, crop2], axis = 3)               # 1/4H * 1/4W * 512                        
			conv3_d = slim.repeat(concat2, 2, slim.conv2d, 256, [3, 3], scope='conv3_d') 

			upsample3 = slim.conv2d_transpose(conv3_d, 128, [4,4], stride=2, scope='upsample3')
			crop3 = tf.slice(upsample3, [0, 0, 0, 0], tf.shape(conv2), name='crop3')
			concat3 = tf.concat([conv2, crop3], axis = 3)               # 1/2H * 1/2W * 256                              
			conv2_d = slim.repeat(concat3, 2, slim.conv2d, 128, [3, 3], scope='conv2_d') 

			upsample4 = slim.conv2d_transpose(conv2_d, 64, [4,4], stride=2, scope='upsample4')
			crop4 = tf.slice(upsample4, [0, 0, 0, 0], tf.shape(conv1), name='crop4')
			concat4 = tf.concat([conv1, crop4], axis = 3)               # H * W * 128
			conv1_d = slim.repeat(concat4, 2, slim.conv2d, 64, [3, 3], scope='conv1_d') 

			end_points = slim.utils.convert_collection_to_dict(end_points_collection)
			logits = slim.conv2d(conv1_d, num_classes, [1,1], activation_fn=None, scope = 'logits')
			
			return logits

if __name__ == '__main__':
	inputs = tf.placeholder(dtype = tf.float32, shape = [4, 224, 224, 3])
	logits = U_Net(inputs = inputs, num_classes = 21)
	print (logits)