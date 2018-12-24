#coding=utf-8
'''
  This is a script used to prepare dataset for training and evaluation, which consists of 
folloing functions:
    1) Read images and labels from TFRecord files; 
    2) Transform images and labels for data augmentation, such as scaling, cropping and fliping;
    3) Subtract image RGB mean values and normalize image;
    4) Assemble images and labels into a mini-batch for training or evalution
'''

import os 
import tensorflow as tf
from preprocess import preprocess
import glob
from PIL import Image
import numpy as np

slim=tf.contrib.slim
tfexample_decoder=slim.tfexample_decoder
dataset_data_provider=slim.dataset_data_provider

IMAGE='image'
ORIGINAL_IMAGE='original_image'
IMAGE_NAME = 'image_name'
HEIGHT='height'
WIDTH='width'
LABEL='label'
_FILE_PATTERN = '%s-*'

def get_dataset(dataset_name, split, dataset_dir, num_samples, num_classes, ignore_label):
  '''Build an instance of slim.dataset.Dataset from TFRecord files

  Args:
    dataset_name: Dataset name (e.g., pascal_voc2012, cityscapes, ADE20K)
    split:  Dataset split (e.g., train, val, test)
    dataset_dir: Dir in which the TFRecord files locates.
    num_samples: The number of samples of the dataset split.
    num_classes: The number of dataset categories.
    ignore_label: The label value which will be ignored for training and evaluation.
  
  Returns:
    A slim.dataset.Dataset object.
  '''
  file_pattern = os.path.join(dataset_dir, _FILE_PATTERN% split)
  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'image/height': tf.FixedLenFeature((), tf.int64, default_value=0),
      'image/width': tf.FixedLenFeature((), tf.int64, default_value=0),
      'image/segmentation/class/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/segmentation/class/format': tf.FixedLenFeature((), tf.string, default_value='png'),
	}
  items_to_handlers = {
      'image': tfexample_decoder.Image(image_key='image/encoded',
                                      format_key='image/format',
                                      channels=3),
      'image_name': tfexample_decoder.Tensor('image/filename'),
      'height': tfexample_decoder.Tensor('image/height'),
      'width': tfexample_decoder.Tensor('image/width'),
      'label': tfexample_decoder.Image(image_key='image/segmentation/class/encoded',
                                        format_key='image/segmentation/class/format',
                                        channels=1),
      }
  decoder = tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=tf.TFRecordReader,
      decoder=decoder,
      num_samples=num_samples,  
      items_to_descriptions = None,    
      ignore_label=ignore_label,
      num_classes=num_classes,
      name=dataset_name,
      multi_label=True)


def get_sample( 
            dataset_name,              
				    split,                     
				    dataset_dir,              
				    train_crop_size,          
            num_classes,               
            num_samples,               
            ignore_label=255,          
        		min_scale_factor=1.,       
        		max_scale_factor=1.,       
        		scale_factor_step_size=0,  
            is_training=True,          
        		num_readers=1,             
        		num_threads=1,        		 
        		batch_size=1,              
            mean_pixel=0 ):            
  '''Read images and labels from TFRecored files, preprocess and assemble them into a mini-batch
    
    Args:
      dataset_name:  Dataset name (e.g., pascal_voc2012, cityscapes, ADE20K)
      split: Dataset split (e.g., train, val, test)
      dataset_dir: Dir in which the TFRecord files locates.
      train_crop_size : A tuple of (crop_heigt, crop_width) used to crop the image and label.
      num_classess: The number of dataset categories.
      num_samples: The number of samples of the dataset split.
      ignore_label: The label value which will be ignored for training and evaluation.
      min_scale_factor:  Mininum scale factor for data augmentation
      max_scale_factor:  Maximum scale factor for data augmentation
      scale_factor_step_size: Scale factor step size for data augmentation
      is_training: Whether to scale, crop and flip image. It is set to True for training and False for evalution. 
      num_readers: Set up how many reader to read TFRecord files. 
      num_threads: Set up how many threads to read TFRecord files. 
      batch_size: The number of samples of a mini-batch.
      mean_pixel: Image RGB mean values.

    Returns:
      A mini-batch (train or validation) that can be fed into network directly.
  '''
  # Build an instance of slim.dataset.Dataset
  dataset = get_dataset(dataset_name, split, dataset_dir, num_samples, num_classes, ignore_label)
  
  # Build a data_provider based on dataset
  data_provider=dataset_data_provider.DatasetDataProvider(
		dataset=dataset,                       # An instance of slim.dataset.
		num_epochs=None if is_training else 1, # The number of times the whole dataset is read repeatedly.
    num_readers=num_readers,               # The number of parallel readers.
		shuffle=is_training		                 # Whether to shuffle dataset.
		)

  # Read image, label, height, witdh and image name from data_provider
  image,image_name,height, width,label = data_provider.get([IMAGE, IMAGE_NAME, HEIGHT, WIDTH, LABEL])

  # Data augmentation including scaling, croping and flip randomly. 
  original_image,image,label=preprocess.process_image_label(
                        image,
												label,
												crop_size=train_crop_size,
												min_scale_factor=min_scale_factor,
												max_scale_factor=max_scale_factor,
												scale_factor_step_size=scale_factor_step_size,
                        ignore_label=ignore_label,
												is_training=is_training,
                        mean_pixel=mean_pixel)

  # Mean_image_subtraction and normalization
  image = tf.cast(image , dtype = tf.float32)
  image -= mean_pixel
  image = (1 / 255.0) * tf.to_float(image)
  sample={
	  IMAGE:image,
	  HEIGHT:height,
	  WIDTH:width,
	  LABEL:label
	}  
  if not is_training:
    sample[ORIGINAL_IMAGE] = original_image
  
  return tf.train.batch(
      sample,
      batch_size=batch_size,
      num_threads=num_threads,
      capacity=32 ,
      allow_smaller_final_batch=not is_training,
      dynamic_pad=True)


# test
if __name__ == '__main__':
  
  dataset='pascl_voc_seg'
  # dataset = 'cityscapes'
  # dataset = 'ADE20K'

  train_split='train'
  val_split = 'val'  

  dataset_dir = "E:/python/Semantic/dataset/baidu_people/tfrecord"
  # dataset_dir='./dataset/pascl_voc_seg/tfrecord'
  # dataset_dir='./dataset/cityscapes/tfrecord'
  # dataset_dir='./dataset/ADE20K/tfrecord'
  
  num_samples =300
  # num_samples = 2975
  # num_samples= 20210  

  IMG_MEAN = np.array((122.67891434,116.66876762,104.00698793), dtype=np.float32)  
  samples=get_sample( 
            dataset_name=dataset, 
            split = val_split, 
            dataset_dir=dataset_dir,
            train_crop_size=[800,600],
            num_classes =2,
            num_samples =num_samples,
            ignore_label = 255,
            min_scale_factor=1,
            max_scale_factor=1.,
            scale_factor_step_size=0.25,
            is_training=False,
            batch_size=3,
            mean_pixel=IMG_MEAN
            )
  
  with tf.Session() as sess:
    
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) 
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)    
    
    samples=sess.run(samples)
    for i in range(3):
      img = (samples[IMAGE][i])* 255 + IMG_MEAN
      img = tf.cast(img,tf.uint8)
      img = Image.fromarray(img.eval(), 'RGB')
      img.show()

      lab=tf.squeeze(samples[LABEL][i])
      lab=tf.cast(lab,tf.uint8)
      lab=Image.fromarray(lab.eval(),'L')
      lab.show()

    coord.request_stop()
    coord.join(threads)