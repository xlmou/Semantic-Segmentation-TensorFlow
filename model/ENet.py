#coding=utf-8
'''
============================================================================
ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation
============================================================================

'''

import tensorflow as tf
slim = tf.contrib.slim


@slim.add_arg_scope
def prelu(x, scope, decoder=False):
    '''
    Performs the parametric relu operation. This implementation is based on:
    https://stackoverflow.com/questions/39975676/how-to-implement-prelu-activation-in-tensorflow

    For the decoder portion, prelu becomes just a normal prelu

    Args:
    - x : a 4D Tensor that undergoes prelu
    - scope: the string to name your prelu operation's alpha variable.
    - decoder: if True, prelu becomes a normal relu.

    Returns:
    - pos + neg / x (Tensor): gives prelu output only during training; otherwise, just return x.

    '''
    #If decoder, then perform relu and just return the output
    if decoder:
        return tf.nn.relu(x, name=scope)

    alpha= tf.get_variable(scope + 'alpha', x.get_shape()[-1],
                           initializer=tf.constant_initializer(0.0),
                           dtype=tf.float32)
    pos = tf.nn.relu(x)
    neg = alpha * (x - abs(x)) * 0.5
    return pos + neg

def spatial_dropout(x, p, seed, scope, is_training=True):
    '''
    Performs a 2D spatial dropout that drops layers instead of individual elements in an input feature map.
    Note that p stands for the probability of dropping, but tf.nn.relu uses probability of keeping.

    ------------------
    Technical Details
    ------------------
    The noise shape must be of shape [batch_size, 1, 1, num_channels], with the height and width set to 1, because
    it will represent either a 1 or 0 for each layer, and these 1 or 0 integers will be broadcasted to the entire
    dimensions of each layer they interact with such that they can decide whether each layer should be entirely
    'dropped'/set to zero or have its activations entirely kept.
    --------------------------

    Args:
    - x(Tensor): a 4D Tensor of the input feature map.
    - p(float): a float representing the probability of dropping a layer
    - seed(int): an integer for random seeding the random_uniform distribution that runs under tf.nn.relu
    - scope(str): the string name for naming the spatial_dropout
    - is_training(bool): to turn on dropout only when training. Optional.

    Returns:
    - A 4D Tensor that is in exactly the same size as the input x,
                      with certain layers having their elements all set to 0 (i.e. dropped).
    '''
    if is_training:
        keep_prob = 1.0 - p
        input_shape = x.get_shape().as_list()
        noise_shape = tf.constant(value=[input_shape[0], 1, 1, input_shape[3]])
        output = tf.nn.dropout(x, keep_prob, noise_shape, seed=seed, name=scope)

        return output

    return x

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

@slim.add_arg_scope
def initial_block(inputs, is_training=True, scope='initial_block'):
	'''The initial block for Enet has 2 branches: The convolution branch and Maxpool branch.
	
	Args:
	  inputs(Tensor): A 4D tensor of shape [batch_size, height, width, channels]
	Returns:
	  A 4D Tensor 
	'''

	initial_conv = slim.conv2d(inputs, 13, [3,3], stride=2, activation_fn=None, scope=scope+'_conv')
	initial_pool = slim.max_pool2d(inputs, [2,2], stride=2, scope=scope+'_pool')
	
	initial_concat = tf.concat([initial_conv, initial_pool], axis=3, name=scope+'_concat')
	initial_concat = slim.batch_norm(initial_concat, is_training=is_training, fused=True, scope=scope+'_batchnorm')
	initial_concat = prelu(initial_concat, scope=scope+'_prelu')
	return initial_concat


@slim.add_arg_scope
def bottleneck(inputs,
               output_depth,
               filter_size,
               regularizer_prob,
               projection_ratio=4,
               seed=0,
               is_training=True,
               downsampling=False,
               upsampling=False,
               pooling_indices=None,
               output_shape=None,
               dilated=False,
               dilation_rate=None,
               asymmetric=False,
               decoder=False,
               scope='bottleneck'):
    '''
    The bottleneck module has three different kinds of variants:
    1. A regular convolution which you can decide whether or not to downsample.
    2. A dilated convolution, which needs to set a dilation factor.
    3. An asymmetric convolution that has a decomposed filter size of 5x1 and 1x5 separately.

    Args:
    - inputs(Tensor): a 4D Tensor of the previous convolutional block of shape [batch_size, height, width, num_channels].
    - output_depth(int): an integer indicating the output depth of the output convolutional block.
    - filter_size(int): an integer that gives the height and width of the filter size to use for a regular/dilated convolution.
    - regularizer_prob(float): the float p that represents the prob of dropping a layer for spatial dropout regularization.
    - projection_ratio(int): the amount of depth to reduce for initial 1x1 projection. Depth is divided by projection ratio. Default is 4.
    - seed(int): an integer for the random seed used in the random normal distribution within dropout.
    - is_training(bool): a boolean value to indicate whether or not is training. Decides batch_norm and prelu activity.

    - downsampling: if True, a max-pool2D layer is added to downsample the spatial sizes.
    - upsampling: if True, the upsampling bottleneck is activated but requires pooling indices to upsample.
    - pooling_indices: the argmax values that are obtained after performing tf.nn.max_pool_with_argmax.
    - output_shape: A list of integers indicating the output shape of the unpooling layer.
    - dilated: if True, then dilated convolution is done, but requires a dilation rate to be given.
    - dilation_rate: the dilation factor for performing atrous convolution/dilated convolution.
    - asymmetric: if True, then asymmetric convolution is done, and the only filter size used here is 5.
    - decoder: if True, then all the prelus become relus according to ENet author.
    - scope: a string name that names your bottleneck.
    Returns:
    - net : The convolution block output after a bottleneck
    - pooling_indices: If downsample, then this tensor is produced for use in upooling later.
    - inputs_shape: The shape of the input to the downsampling conv block. For use in unpooling later.

    '''
    # Calculate the depth reduction based on the projection ratio used in 1x1 convolution.
    reduced_depth = int(output_depth / projection_ratio)
    inputs_shape = tf.shape(inputs)
    with slim.arg_scope([prelu], decoder=decoder):

        # Downsampling bottleneck
        if downsampling:
            # Main brach: perform a max pooling, a convolution and a batch norm
            net_main, pooling_indices = tf.nn.max_pool_with_argmax(inputs,
                                                                   ksize=[1,2,2,1],
                                                                   strides=[1,2,2,1],
                                                                   padding='SAME',
                                                                   name=scope+'_main_max_pool')
            net_main = slim.conv2d(net_main, output_depth, [1,1], scope=scope+'_main_conv')
            net_main = slim.batch_norm(net_main, is_training=is_training, scope=scope+'_main_batch_norm')
 
            # Sub branch: first projection that has a 2x2 kernel and stride 2
            net = slim.conv2d(inputs, reduced_depth, [2,2], stride=2, scope=scope+'_conv1')
            net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm1')
            net = prelu(net, scope=scope+'_prelu1')

            # Sub branch: second conv block
            net = slim.conv2d(net, reduced_depth, [filter_size, filter_size], scope=scope+'_conv2')
            net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm2')
            net = prelu(net, scope=scope+'_prelu2')

            # Sub branch: final projection with 1x1 kernel
            net = slim.conv2d(net, output_depth, [1,1], scope=scope+'_conv3')
            net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm3')
            net = prelu(net, scope=scope+'_prelu3')
            net = spatial_dropout(net, p=regularizer_prob, seed=seed, scope=scope+'_spatial_dropout')
            
            # Concatenate main branch and sub branch
            net = tf.add(net, net_main, name=scope+'_add')
            net = prelu(net, scope=scope+'_last_prelu')
            return net, pooling_indices, inputs_shape

        #  Dilation convolution bottleneck
        elif dilated:            
            if not dilation_rate:
                raise ValueError('Dilation rate is not given.')

            # Save the main branch for addition later
            net_main = inputs

            #First projection with 1x1 kernel (dimensionality reduction)
            net = slim.conv2d(inputs, reduced_depth, [1,1], scope=scope+'_conv1')
            net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm1')
            net = prelu(net, scope=scope+'_prelu1')

            #Second conv block --- apply dilated convolution here
            net = slim.conv2d(net, reduced_depth, [filter_size, filter_size], rate=dilation_rate, scope=scope+'_dilated_conv2')
            net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm2')
            net = prelu(net, scope=scope+'_prelu2')

            #Final projection with 1x1 kernel (Expansion)
            net = slim.conv2d(net, output_depth, [1,1], scope=scope+'_conv3')
            net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm3')
            net = prelu(net, scope=scope+'_prelu3')
            net = spatial_dropout(net, p=regularizer_prob, seed=seed, scope=scope+'_spatial_dropout')

            #Add the main branch
            net = tf.add(net_main, net, name=scope+'_add_dilated')
            net = prelu(net, scope=scope+'_last_prelu')
            return net

        # Asymmetric convolution bottleneck     
        elif asymmetric:
            net_main = inputs

            #First projection with 1x1 kernel (dimensionality reduction)
            net = slim.conv2d(inputs, reduced_depth, [1,1], scope=scope+'_conv1')
            net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm1')
            net = prelu(net, scope=scope+'_prelu1')

            # Second conv block --- apply asymmetric conv here
            net = slim.conv2d(net, reduced_depth, [filter_size, 1], scope=scope+'_asymmetric_conv2a')
            net = slim.conv2d(net, reduced_depth, [1, filter_size], scope=scope+'_asymmetric_conv2b')
            net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm2')
            net = prelu(net, scope=scope+'_prelu2')

            # Final projection with 1x1 kernel
            net = slim.conv2d(net, output_depth, [1,1], scope=scope+'_conv3')
            net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm3')
            net = prelu(net, scope=scope+'_prelu3')
            net = spatial_dropout(net, p=regularizer_prob, seed=seed, scope=scope+'_spatial_dropout')

            #Add the main branch
            net = tf.add(net_main, net, name=scope+'_add_asymmetric')
            net = prelu(net, scope=scope+'_last_prelu')
            return net

        # Upsampling bottleneck
        elif upsampling:
            if pooling_indices == None:
                raise ValueError('Pooling indices are not given.')
            if output_shape == None:
                raise ValueError('Output depth is not given')

            # Main branch
            net_unpool = slim.conv2d(inputs, output_depth, [1,1], scope=scope+'_main_conv1')
            net_unpool = slim.batch_norm(net_unpool, is_training=is_training, scope=scope+'batch_norm1')
            net_unpool = unpool(net_unpool, pooling_indices, output_shape=output_shape, scope='unpool')

            # Sub branch: first 1x1 projection to reduce depth
            net = slim.conv2d(inputs, reduced_depth, [1,1], scope=scope+'_conv1')
            net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm2')
            net = prelu(net, scope=scope+'_prelu1')

            # Sub branch: Second conv block 
            net = slim.conv2d_transpose(net, reduced_depth, [4, 4], stride=2, scope=scope+'_transposed_conv2')
            net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm3')
            net = prelu(net, scope=scope+'_prelu2')

            # Final projection with 1x1 kernel
            net = slim.conv2d(net, output_depth, [1,1], scope=scope+'_conv3')
            net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm4')
            net = prelu(net, scope=scope+'_prelu3')
            net = spatial_dropout(net, p=regularizer_prob, seed=seed, scope=scope+'_spatial_dropout')

            # Finally, add the unpooling layer and the sub branch together
            net = tf.add(net, net_unpool, name=scope+'_add_upsample')
            net = prelu(net, scope=scope+'_last_prelu')
            return net

        # Regular convolution bottleneck
        net_main = inputs

        # First projection with 1x1 kernel
        net = slim.conv2d(inputs, reduced_depth, [1,1], scope=scope+'_conv1')
        net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm1')
        net = prelu(net, scope=scope+'_prelu1')

        # Second conv block
        net = slim.conv2d(net, reduced_depth, [filter_size, filter_size], scope=scope+'_conv2')
        net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm2')
        net = prelu(net, scope=scope+'_prelu2')

        # Final projection with 1x1 kernel
        net = slim.conv2d(net, output_depth, [1,1], scope=scope+'_conv3')
        net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm3')
        net = prelu(net, scope=scope+'_prelu3')
        net = spatial_dropout(net, p=regularizer_prob, seed=seed, scope=scope+'_spatial_dropout')

        # Add the main branch
        net = tf.add(net_main, net, name=scope+'_add_regular')
        net = prelu(net, scope=scope+'_last_prelu')
        return net


def ENet(inputs, num_classes, is_training=True):
    '''The ENet model for real-time semantic segmentation based on 
       https://github.com/TimoSaemann/ENet

    Args:
    - inputs: a 4D Tensor of shape [batch_size, height, width, num_channels].
    - num_classes: an integer for the number of classes to predict.
    - is_training: if True, switch on batch_norm and prelu only during training, otherwise they are turned off.
    Returns:
    - A 4D Tensor output of shape [batch_size, height, width, num_classes]
    '''

    with tf.variable_scope('ENet'):
        with slim.arg_scope([initial_block, bottleneck], is_training=is_training),\
             slim.arg_scope([slim.batch_norm], fused=True), \
             slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=None): 

            # Initial block
            net_one = net = initial_block(inputs, scope='initial_block_1')

            # Stage one
            net, pooling_indices_1, inputs_shape_1 = bottleneck(net,output_depth=64, filter_size=3,regularizer_prob=0.01,downsampling=True,scope='bottleneck1_0')
            net = bottleneck(net, output_depth=64, filter_size=3, regularizer_prob=0.01, scope='bottleneck1_1')
            net = bottleneck(net, output_depth=64, filter_size=3, regularizer_prob=0.01, scope='bottleneck1_2')
            net = bottleneck(net, output_depth=64, filter_size=3, regularizer_prob=0.01, scope='bottleneck1_3')
            net = bottleneck(net, output_depth=64, filter_size=3, regularizer_prob=0.01, scope='bottleneck1_4')

            # Stage two and stage three, regularization prob is 0.1 from bottleneck 2.0 onwards
            with slim.arg_scope([bottleneck], regularizer_prob=0.1):
                net, pooling_indices_2, inputs_shape_2 = bottleneck(net, output_depth=128, filter_size=3, downsampling=True, scope='bottleneck2_0')                
                for i in range(2, 4):
                    net = bottleneck(net, output_depth=128, filter_size=3, scope='bottleneck'+str(i)+'_1')
                    net = bottleneck(net, output_depth=128, filter_size=3, dilated=True, dilation_rate=2, scope='bottleneck'+str(i)+'_2')
                    net = bottleneck(net, output_depth=128, filter_size=5, asymmetric=True, scope='bottleneck'+str(i)+'_3')
                    net = bottleneck(net, output_depth=128, filter_size=3, dilated=True, dilation_rate=4, scope='bottleneck'+str(i)+'_4')
                    net = bottleneck(net, output_depth=128, filter_size=3, scope='bottleneck'+str(i)+'_5')
                    net = bottleneck(net, output_depth=128, filter_size=3, dilated=True, dilation_rate=8, scope='bottleneck'+str(i)+'_6')
                    net = bottleneck(net, output_depth=128, filter_size=5, asymmetric=True, scope='bottleneck'+str(i)+'_7')
                    net = bottleneck(net, output_depth=128, filter_size=3, dilated=True, dilation_rate=16, scope='bottleneck'+str(i)+'_8')
            
            with slim.arg_scope([bottleneck], regularizer_prob=0.1, decoder=True):           
             
                # Stage four
                bottleneck_scope_name = "bottleneck" + str(i + 1)
                net = bottleneck(net, output_depth=64, 
                				 filter_size=3, upsampling=True,
                                 pooling_indices=pooling_indices_2, 
                                 output_shape=inputs_shape_2, 
                                 scope=bottleneck_scope_name+'_0')              
                net = bottleneck(net, output_depth=64, filter_size=3, scope=bottleneck_scope_name+'_1')
                net = bottleneck(net, output_depth=64, filter_size=3, scope=bottleneck_scope_name+'_2')

                # Stage five
                bottleneck_scope_name = "bottleneck" + str(i + 2)
                net = bottleneck(net, output_depth=16, 
                				 filter_size=3, upsampling=True,
                                 pooling_indices=pooling_indices_1, 
                                 output_shape=inputs_shape_1, 
                                 scope=bottleneck_scope_name+'_0')                
                net = bottleneck(net, output_depth=16, filter_size=3, scope=bottleneck_scope_name+'_1')

            logits = slim.conv2d_transpose(net, num_classes, [2,2], stride=2, scope='fullconv')

        return logits


def ENet_arg_scope(weight_decay=2e-4,
                   batch_norm_decay=0.1,
                   batch_norm_epsilon=0.001):
  '''The arg scope for enet model. 

  Args:
  - weight_decay: the weight decay for weights variables in conv2d and separable conv2d
  - batch_norm_decay: decay for the moving average of batch_norm momentums.
  - batch_norm_epsilon: small float added to variance to avoid dividing by zero.

  Returns:
  - A tf-slim arg_scope with the parameters needed for xception.
  '''
  # Set weight_decay for weights in conv2d and separable_conv2d layers.
  with slim.arg_scope([slim.conv2d],
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_regularizer=slim.l2_regularizer(weight_decay)):

    # Set parameters for batch_norm.
    with slim.arg_scope([slim.batch_norm],
                        decay=batch_norm_decay,
                        epsilon=batch_norm_epsilon) as scope:
      return scope

if __name__ == '__main__':
	inputs = tf.placeholder(dtype = tf.float32, shape = [4, 224, 224, 3])
	logits = ENet(inputs = inputs, num_classes = 21, is_training=True)	
	print (logits)
