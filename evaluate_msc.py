"""
Multi-scale evaluation script for the semantic segmentation network on the validation subset
of pascal voc2012, cityscapes and ADE20K.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

import tensorflow as tf
import numpy as np

from build_model import build_model
from utils import decode_labels
from utils import inv_preprocess
from utils import prepare_label
from preprocess import seg_dataset

DATASET_LIST = ["pascal_voc2012", "cityscapes", "ADE20K"]
SEGMENTATION_MODEL_LIST = ["FCN8", "U_Net", "Seg_Net","Deeplab_v1", "Deeplab_v2", "Deeplab_v3", 
                            "PSPNet", "GCN", "ENet", "ICNet"]
DATASET_TO_CLASSES = {"pascal_voc2012":21, "cityscapes":19, "ADE20K":151}
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = '/dataset/tfrecord'             # Path to the directory containing dataset.
DATA_SET = 'pascal_voc_2012'                     # Dataset name
VAL_SPLIT = 'val'                                # The split used to evaluation
IGNORE_LABEL = 255                               # Ignore label
NUM_VAL_SAMPLES = 1449                           # The number of validation set
RESTORE_FROM = None                              # Path to pretrained model 
SEGMENTATION_MODEL = "FCN8"                      # Semantic segmentation model

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Semantic Segmentation Network")
    parser.add_argument("--dataset", type=str, default=DATA_SET, 
                                help="the dataset name ")    
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                                help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--val-split", type=str, default=VAL_SPLIT,
                                help="the split used to val")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                                help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                                help="Number of classes to predict (including background).")
    parser.add_argument("--num-val-samples", type=int, default=NUM_VAL_SAMPLES,
                                help="Number of images in the validation set.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                                help="Where restore model parameters from.")
    parser.add_argument("--segmentation-model", type=str, default=SEGMENTATION_MODEL, 
                                    help="Semantic segmentation model name.") 
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments() 
    
    # check out dataset
    if (args.dataset not in DATASET_LIST):
        print ("The dataset is not supported")
        return False
    
    # check out segmentation model
    if (args.segmentation_model not in SEGMENTATION_MODEL_LIST):
        print ("The segmentation model is not supported")
        return False

    num_classes = DATASET_TO_CLASSES[args.dataset]   
    
    # Load evaluation batch
    print ("load data...")
    val_samples=seg_dataset.get_sample( 
                                    dataset_name=args.dataset, 
                                    split=args.val_split, 
                                    dataset_dir=args.data_dir,
                                    train_crop_size=None,
                                    num_classes=num_classes,
                                    num_samples=args.num_val_samples,
                                    ignore_label=args.ignore_label,                                    
                                    min_scale_factor=1,
                                    max_scale_factor=1,
                                    scale_factor_step_size=0,
                                    is_training=False,
                                    batch_size=1,
                                    mean_pixel=IMG_MEAN)    

    print ("create network...")
    image_batch = val_samples['image']
    label_batch = val_samples['label']

    h_orig, w_orig = tf.to_float(tf.shape(image_batch)[1]), tf.to_float(tf.shape(image_batch)[2])
    image_batch075 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.75)), tf.to_int32(tf.multiply(w_orig, 0.75))]))
    image_batch050 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.5)), tf.to_int32(tf.multiply(w_orig, 0.5))]))

    with tf.variable_scope('', reuse=False):
        output100 = build_model(inputs = image_batch, num_classes = num_classes, segmentation_model = args.segmentation_model, is_training = False)
    with tf.variable_scope('', reuse=True):
        output075 = build_model(inputs = image_batch075, num_classes = num_classes, segmentation_model = args.segmentation_model, is_training = False)
    with tf.variable_scope('', reuse=True):
        output050 = build_model(inputs = image_batch050, num_classes = num_classes, segmentation_model = args.segmentation_model, is_training = False)

    # Which variables to load.
    restore_var = tf.global_variables()
    
    # Predictions.
    output075 = tf.image.resize_images(output075, tf.shape(output100)[1:3])
    output050 = tf.image.resize_images(output050, tf.shape(output100)[1:3])
    
    pred_img = tf.reduce_max(tf.stack([output100, output075, output050]), axis=0)
    pred_img = tf.image.resize_bilinear(pred_img, tf.shape(image_batch)[1:3])
    pred_img = tf.argmax(pred_img, axis=3)
    pred_img = tf.expand_dims(pred_img, axis=3) 
    
    # comput mean Intersection over Union
    pred_vector = tf.reshape(pred_img, [-1,])
    label_vector = tf.reshape(label_batch, [-1,])

    # ignore all labels >= num_classes
    indices = tf.squeeze(tf.where(tf.less_equal(label_vector, num_classes - 1)), 1)  
    label_vector = tf.cast(tf.gather(label_vector, indices), tf.int32)
    pred_vector = tf.gather(pred_vector, indices)
    mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred_vector, label_vector, num_classes=num_classes)    
        
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())    
    sess.run(init)
    
    # create a loader to save or restore model
    loader = tf.train.Saver(var_list=restore_var)
    if args.restore_from is not None:
        load(loader, sess, args.restore_from)
    
    # Before loading data, we need to create coordinator and start queue runner.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)    
  
    for step in range(args.num_val_samples):
        print ("step:",step)
        prediction, _ = sess.run([pred_img, update_op])
        if step % 100 == 0:
            print('step {:d}'.format(step))
    print('Mean IoU: {:.3f}'.format(mIoU.eval(session=sess)))
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()