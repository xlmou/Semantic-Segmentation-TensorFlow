#coding = utf-8
'''
============================================================================
ICNet for Real-Time Semantic Segmentation on High-Resolution Images
============================================================================

  In order to replicate the training procedure in the original ICNet paper, multiple steps 
must be taken. In particular, transfer learning must be done from the baseline PSPNet50 
model. Compression must also then be used. 
    Stage 0 ~ Pre-train a PSPNet50 model: First, a PSPNet50 model is trained on weights 
initialized from a dilated ResNet50 model. Using a similar training procedure as described
in the original paper (with a crop size of 768, 120K training iterations and an initial 
learning rate of 0.01), the PSPNet50 model in this project was trained and converged at 
approximately 74% mIoU.
    Stage 1 ~ Initialize ICNet Branch from PSPNet50: With a base PSPNet50 model trained, 
the second stage of training can begin by initializing the ICNet quarter resolution 
branch with the pre-trained PSPNet50 model (with a crop size of 1024, 200K training 
iterations and an initial learning rate of 0.001). Initializing ICNet from these weights 
allowed for convergence at accuracies similar to the original ICNet paper.
    Stage 2 ~ Compression and Retraining: Once the base ICNet model is trained, we must prune 
half of the kernels to achieve the performance of the original paper. This is a process 
where kernels are removed from each convolutional layer iteratively. After the kernels 
are pruned, the pruned model must be retrained a final time to recover from the lost 
accuracy during pruning.

  More detials are described in 
https://github.com/oandrienko/fast-semantic-segmentation/blob/master/docs/icnet.md
'''
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.nets import resnet_v2

slim = tf.contrib.slim


def Sampling(inputs, scale = 2):
    shape = tf.to_float(tf.shape(inputs)[1:3])
    shape = tf.to_int32(tf.multiply(shape, scale))
    return tf.image.resize_bilinear(inputs, size=shape)


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



def PSPNet(inputs , num_classes, is_training, reuse=None):
	'''A TensorFlow implementation of PSPNet model based on 
	   https://github.com/hszhao/PSPNet/tree/master/evaluation/prototxt	   	
	
	Args:
		inputs: A 4-D tensor with dimensions [batch_size, height, width, channels]
		num_classes: Integer, the total number of categories in the dataset
		is_training : Bool, whether to updates the running means and variances during the training.
	Returns:
		A feature map with dimensions [batch_size, 1/8*height, 1/8*width, 512]

	'''
	with slim.arg_scope(resnet_utils.resnet_arg_scope()):
		net, end_points = resnet_v2.resnet_v2_50(inputs,
							                      num_classes=None,
							                      is_training=is_training,
							                      global_pool=False,
							                      output_stride=8,
							                      reuse=reuse,
							                      scope='resnet_v2_50')
	with tf.variable_scope("PSPNet", reuse=reuse) as sc:    	
		shape = tf.shape(net)[1:3]
		net = PyramidPoolingModule(net, shape = shape)
		net = slim.conv2d(net, 512, [3, 3], activation_fn=None, scope='conv5')
		net = slim.batch_norm(net, fused=True, scope='conv5_bn')
		net = tf.nn.relu(net, name='conv5_bn_relu')
	return net,end_points


def ICNet(inputs , num_classes, is_training):
	'''A TensorFlow implementation of ICNet evaluate model(not train model) based on 
	   https://github.com/hszhao/ICNet/blob/master/evaluation/prototxt/icnet_cityscapes.prototxt   	
	
	Args:
		inputs: A 4-D tensor with dimensions [batch_size, height, width, channels]
		num_classes: Integer, the total number of categories in the dataset
		is_training : Bool, whether to updates the running means and variances during the training.
	Returns:
		A score map with dimensions [batch_size, 1/4*height, 1/4*width, num_classes]

	'''
	
	subsample4 = Sampling(inputs, scale=1/4)
	branch1, _ = PSPNet(subsample4, num_classes=num_classes, is_training=is_training)

	subsample2 = Sampling(inputs, scale=1/2)
	_,end_points = PSPNet(subsample2, num_classes=num_classes, is_training=is_training, reuse=True)
	branch2 = end_points['resnet_v2_50/block2/unit_1/bottleneck_v2']  
	
	with tf.variable_scope('ENet'):
		branch1 = Sampling(branch1, scale=2)
		branch1 = slim.conv2d(branch1, 256, [3,3], rate=2, activation_fn=None, 
							  scope='branch1_conv')
		branch1 = slim.batch_norm(branch1, fused=True)

		branch2 = slim.conv2d(branch2, 256, [1,1], activation_fn=None, scope='branch2_conv')
		branch2 = slim.batch_norm(branch2, fused=True)

		# fuse branch2 and branch1
		fused1 = tf.add(branch1, branch2, name='fused1')
		fused1 = tf.nn.relu(fused1)
		fused1 = Sampling(fused1, scale=2)
		fused1 = slim.conv2d(fused1, 256, [3,3], rate=2, activation_fn=None, scope='fused1_conv')
		fused1 = slim.batch_norm(fused1, fused=True)

		# prepare branch3
		branch3 = slim.conv2d(inputs, 64, [3,3], stride=2, activation_fn=None, scope='branch3_conv1')
		branch3 = slim.batch_norm(branch3, fused=True)
		branch3 = tf.nn.relu(branch3)

		branch3 = slim.conv2d(branch3, 64, [3,3], stride=2, activation_fn=None, scope='branch3_conv2')
		branch3 = slim.batch_norm(branch3, fused=True)
		branch3 = tf.nn.relu(branch3)

		branch3 = slim.conv2d(branch3, 64, [3,3], stride=2, activation_fn=None, scope='branch3_conv3')
		branch3 = slim.batch_norm(branch3, fused=True)
		branch3 = tf.nn.relu(branch3)

		branch3 = slim.conv2d(branch3, 256, [1,1], activation_fn=None, scope='branch3_conv4')
		branch3 = slim.batch_norm(branch3, fused=True)

		# fuse branch3 and branch2 and branch1
		fused2 = tf.add(fused1, branch3, name='fused3')
		fused2 = tf.nn.relu(fused2)
		fused2 = Sampling(fused2, scale=2)
		logits = slim.conv2d(fused2, num_classes, [1,1], activation_fn=None, scope='logits')
		return logits



if __name__ == '__main__':
	inputs = tf.placeholder(dtype = tf.float32, shape = [4, 224, 224, 3])
	logits = ICNet(inputs, num_classes=21, is_training=True)
	print (logits)