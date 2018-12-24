#coding=utf-8
'''
 This script takes responsibility for preprocessing image and label 
 (e.g., scaling, padding, cropping and flipping) for data augmentation.
'''

import tensorflow as tf
from PIL import Image
import numpy as np

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def process_image_label(image,
						label,
						crop_size,
						min_scale_factor=1.,
						max_scale_factor=1.,
						scale_factor_step_size=0,
						ignore_label=255,
						is_training=True,
						mean_pixel=IMG_MEAN):
	'''Scaling, cropping, padding and flipping image and label for data augmentation.
	
	Args:
	  image: An image of shape [image_heigt, image_width, 3] and of type uint8.
	  label: An annotations of shape [image_heigt, image_width, 1] and of type uint8.
	  crop_size: A tuple of (crop_height, crop_width). If None, return original image and label.
	  min_scale_factor: Mininum scale factor.
	  max_scale_factor: Maximum scale factor.
	  scale_factor_step_size: Scale factor step size. The input is randomly scaled based on
	    the value of (min_scale_factor, max_scale_factor, scale_factor_step_size).
	  ignore_label: The label value which will be ignored for training and evaluation.
	  is_training: If true, scale, crop and flip image. If false, pad image and label to crop size
	    when size of the original image is smaller than crop size, otherwise return orginal image and label.
	  mean_pixel: Image size of the mean values.
	
	Returns:
	  Processed image of type float32 and label of type uint8. 
	'''

	processed_image=tf.cast(image, dtype=tf.float32)
	original_image=processed_image

	if crop_size is None:
		return original_image,processed_image,label

	label=tf.cast(label, dtype=tf.float32)

	# If the size of the original image is larger than crop size, then scale image and label , 
	# otherwise return original image and label
	processed_image,label=shrink_image_and_label(processed_image,label,crop_size[0],crop_size[1])

	# Scale image and label randomly based on scale factors.
	if is_training:
		scale = get_random_scale(min_scale_factor,max_scale_factor,scale_factor_step_size)
		processed_image,label = randomly_scale_image_and_label(processed_image,label,scale)
		processed_image.set_shape([None,None,3])

	# Pad image with RGB mean values and pad label with ignore label.
	image_shape=tf.shape(processed_image)
	image_height=image_shape[0]	
	image_width=image_shape[1]	
	target_height = image_height + tf.maximum(crop_size[0] - image_height, 0)
	target_width = image_width + tf.maximum(crop_size[1] - image_width, 0) 		

	if mean_pixel is None:
		mean_pixel=IMG_MEAN
	mean_pixel=tf.reshape(mean_pixel,[1,1,3])
	processed_image = pad_to_bounding_box(processed_image, 0, 0, target_height, target_width, mean_pixel)
	label = pad_to_bounding_box(label, 0, 0, target_height, target_width, ignore_label)

	# Crop image and label to fixed crop size.
	if is_training:
		processed_image, label = random_crop([processed_image, label], crop_size[0], crop_size[1])

	processed_image.set_shape([crop_size[0], crop_size[1], 3])
	label.set_shape([crop_size[0], crop_size[1], 1])
	
	# Flip image and label randomly.
	if is_training:
		processed_image, label, _ = flip_dim([processed_image, label], 0.5, dim=1)

	label = tf.cast(label,tf.uint8)

	return original_image, processed_image, label


def shrink_image_and_label(image,label,height,width):
	'''If the size of orginal image is larger than crop size, then resize image or label 
	   so their sides are within the provided size, otherwise return original image and label

	Args:
	  image: An image of shape [image_heigt, image_width, 3] and of type float32.
	  label: An annotation of shape [image_heigt, image_width, 3] and of type float32.
	  height: crop height
	  width:  crop width

	Returns:
	  Scaled image and label.

	'''
	
	image_shape = tf.shape(image)
	orig_height = tf.to_float(image_shape[0])
	orig_width = tf.to_float(image_shape[1])	

	orig_image = image
	orig_label = label

	# calculate target size 
	height_ratio = orig_height/height
	width_ratio = orig_width/width
	scale1 = [height,tf.to_int32(tf.floor(orig_width * tf.to_float(height / orig_height)))]
	scale2 = [tf.to_int32(tf.floor(orig_height * tf.to_float(width / orig_width))),width]
	target_scale = tf.cond(tf.greater_equal(height_ratio,width_ratio),lambda:scale1,lambda:scale2)

	image = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(image, 0),
						target_scale,align_corners=True),[0])
	label = tf.squeeze(tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0),
							target_scale,align_corners=True), [0])
	
	return tf.cond(tf.logical_or(tf.greater_equal(orig_height, height),
								  tf.greater_equal(orig_width, width)),lambda:(image,label),lambda:(orig_image,orig_label))


def get_random_scale(min_scale_factor, max_scale_factor, step_size):
	"""
	  Pick up a scale factor randomly from (min_scale_factor, max_scale_factor, step_size)
	"""

	if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
		raise ValueError('Unexpected value of min_scale_factor.')
	if min_scale_factor == max_scale_factor:
		return tf.to_float(min_scale_factor)
	if step_size == 0:
		return tf.random_uniform([1],minval=min_scale_factor,maxval=max_scale_factor)

	num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
	scale_factors = tf.lin_space(min_scale_factor, max_scale_factor, num_steps)
	shuffled_scale_factors = tf.random_shuffle(scale_factors)
	return shuffled_scale_factors[0]	

def randomly_scale_image_and_label(image, label=None, scale=1.0):
	"""
	  Scale image and label based on given scale factor
	"""

	if scale == 1.0:
		return image, label

	image_shape = tf.shape(image)
	new_dim = tf.to_int32(tf.to_float( [ image_shape[0], image_shape[1] ] ) * scale)

	image = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(image, 0),
						new_dim,align_corners=True),[0])
	label = tf.squeeze(tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0),
							new_dim,align_corners=True), [0])

	return image, label

def pad_to_bounding_box(image,offset_height,offset_width,target_height,target_width,pad_value):
	"""Padding image and label to target size

	Args:
	  image:  An image of shape [image_heigt, image_width, 3] and of type float32.	  
	  offset_height: Number of rows of zeros to add on top.
	  offset_width:  Number of columns of zeros to add on the left.
	  target_heigt:  Height of output image.
	  target_width:  Width of output image.
	  pad_values:  Value to pad the image tensor with.

	Returns:
	  3-D tensor of shape [target_height, target_width, 3].
	"""
	image_rank = tf.rank(image)
	image_rank_assert = tf.Assert(tf.equal(image_rank, 3),
									['Wrong image tensor rank [Expected] [Actual]',3, image_rank])
	
	# Firstly subtract padding values before padding. 
	with tf.control_dependencies([image_rank_assert]):
		image -= pad_value
	
	# Next calculate padding shape.
	image_shape = tf.shape(image)
	height, width = image_shape[0], image_shape[1]
	target_width_assert = tf.Assert(tf.greater_equal(target_width, width), 
									['target_width must be >= width'])
	target_height_assert = tf.Assert(tf.greater_equal(target_height, height),
									['target_height must be >= height'])
	
	with tf.control_dependencies([target_width_assert]):
		after_padding_width = target_width - offset_width - width
	
	with tf.control_dependencies([target_height_assert]):
		after_padding_height = target_height - offset_height - height
	
	offset_assert = tf.Assert(tf.logical_and(tf.greater_equal(after_padding_width, 0),
											tf.greater_equal(after_padding_height, 0)),
							['target size not possible with the given target offsets'])

	height_params = tf.stack([offset_height, after_padding_height])
	width_params = tf.stack([offset_width, after_padding_width])
	channel_params = tf.stack([0, 0])
	
	with tf.control_dependencies([offset_assert]):
		paddings = tf.stack([height_params, width_params, channel_params])
	
	# Then pad image with 0.
	padded = tf.pad(image, paddings)
	
	# Finally add padding values after padding
	return tf.add(padded,pad_value)


def random_crop(image_list, crop_height, crop_width):
	"""
	  Crop image and label to fixed crop size.
	"""
	if not image_list:
		raise ValueError('Empty image_list.')

	rank_assertions = []
	for i in range(len(image_list)):
		image_rank = tf.rank(image_list[i])
		rank_assert = tf.Assert(tf.equal(image_rank, 3),
        					['Wrong rank for tensor  %s [expected] [actual]',
         					image_list[i].name, 3, image_rank])
		rank_assertions.append(rank_assert)

	with tf.control_dependencies([rank_assertions[0]]):
		image_shape = tf.shape(image_list[0])
	image_height = image_shape[0]
	image_width = image_shape[1]

	crop_size_assert = tf.Assert(
      tf.logical_and(
          tf.greater_equal(image_height, crop_height),
          tf.greater_equal(image_width, crop_width)),
      ['Crop size greater than the image size.'])

	asserts = [rank_assertions[0], crop_size_assert]

	for i in range(1, len(image_list)):
		image = image_list[i]
		asserts.append(rank_assertions[i])
		with tf.control_dependencies([rank_assertions[i]]):
			shape = tf.shape(image)
		height = shape[0]
		width = shape[1]

		height_assert = tf.Assert(tf.equal(height, image_height),
								['Wrong height for tensor %s [expected][actual]',
								image.name, height, image_height])
		width_assert = tf.Assert(tf.equal(width, image_width),
								['Wrong width for tensor %s [expected][actual]',
								image.name, width, image_width])
		asserts.extend([height_assert, width_assert])

	with tf.control_dependencies(asserts):
		max_offset_height = tf.reshape(image_height - crop_height + 1, [])
		max_offset_width = tf.reshape(image_width - crop_width + 1, [])
	offset_height = tf.random_uniform([], maxval=max_offset_height, dtype=tf.int32)
	offset_width = tf.random_uniform([], maxval=max_offset_width, dtype=tf.int32)

	return [_crop(image,offset_height, offset_width,crop_height,crop_width) for image in image_list]


def _crop(image, offset_height, offset_width, crop_height, crop_width):
	'''Crops the given image using the provided offsets and sizes.

	  Note that the method doesn't assume we know the input image size but it does
	  assume we know the input image rank.

	Args:
	    image: an image of shape [height, width, channels].
	    offset_height: a scalar tensor indicating the height offset.
	    offset_width: a scalar tensor indicating the width offset.
	    crop_height: the height of the cropped image.
	    crop_width: the width of the cropped image.

	Returns:
	    The cropped (and resized) image.
	'''

	original_shape = tf.shape(image)

	if len(image.get_shape().as_list()) != 3:
		raise ValueError('input must have rank of 3')
	original_channels = image.get_shape().as_list()[2]

	rank_assertion = tf.Assert(tf.equal(tf.rank(image), 3),
								['Rank of image must be equal to 3.'])
	with tf.control_dependencies([rank_assertion]):
		cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

	size_assertion = tf.Assert(tf.logical_and(
						tf.greater_equal(original_shape[0], crop_height),
						tf.greater_equal(original_shape[1], crop_width)),
						['Crop size greater than the image size.'])

	offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

	with tf.control_dependencies([size_assertion]):
		image = tf.slice(image, offsets, cropped_shape)
	image = tf.reshape(image, cropped_shape)
	image.set_shape([crop_height, crop_width, original_channels])
	return image


def flip_dim(tensor_list, prob=0.5, dim=1):
	"""
	  Randomly flips a dimension of the given tensor.
	"""
	random_value = tf.random_uniform([])
	def flip():
		flipped = []
		for tensor in tensor_list:
			if dim < 0 or dim >= len(tensor.get_shape().as_list()):
				raise ValueError('dim must represent a valid dimension.')
			flipped.append(tf.reverse_v2(tensor, [dim]))
		return flipped

	is_flipped = tf.less_equal(random_value, prob)
	outputs = tf.cond(is_flipped, flip, lambda: tensor_list)
	if not isinstance(outputs, (list, tuple)):
		outputs = [outputs]
	outputs.append(is_flipped)

	return outputs

# test
if __name__ == '__main__': 

    IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
    image= tf.gfile.FastGFile("D:/test/00001.jpg", 'rb').read()
    label=tf.gfile.FastGFile("D:/test/00001.png",'rb').read()
    img_raw = tf.image.decode_jpeg(image)      
    label_raw=tf.image.decode_png(label)
    _,pro_image,pro_label=process_image_label(
	    				img_raw,
						label_raw,
						crop_size=[800,500],
						min_scale_factor=1.0,
						max_scale_factor=1.0,
						scale_factor_step_size=0.1,
						ignore_label=255,
						is_training=True,
						mean_pixel=None)      
    with tf.Session() as sess:
      img_raw = sess.run(pro_image)
      label_raw = sess.run(pro_label)     
      image,label=pro_image,pro_label
      print (image.shape)
      print (label.shape) 
      show_img=tf.cast(image,tf.uint8)
      show_img = Image.fromarray(show_img.eval(), 'RGB')    
      show_img.show()  
      show_label=tf.cast(label,tf.uint8)
      show_label=tf.squeeze(label)       
      show_label= Image.fromarray(show_label.eval(), 'L') 
      show_label.show()