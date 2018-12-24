#coding=utf-8
"""
This is a training script for various semantic segmentation networks on three datasets, including
pascal voc2012, cityscapes and ADE20K. 
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time
import tensorflow as tf
import numpy as np

from utils import decode_labels
from utils import inv_preprocess
from utils import prepare_label
from preprocess import seg_dataset
from build_model import build_model

DATASET_LIST = ["pascal_voc2012", "cityscapes", "ADE20K"]
SEGMENTATION_MODEL_LIST = ["FCN8", "U_Net", "Seg_Net","Deeplab_v1", "Deeplab_v2", 
                            "Deeplab_v3", "PSPNet", "GCN", "ENet", "ICNet"]
DATASET_TO_CLASSES = {"pascal_voc2012":21, "cityscapes":19, "ADE20K":151}
IMG_MEAN = np.array((122.67891434,116.66876762,104.00698793), dtype=np.float32)  # RGB mean values
DATA_DIRECTORY = './dataset/tfrecord/'             # Path to the directory containing dataset.
DATA_SET = "pascal_voc2012"                        # Dataset name
TRAIN_SPLIT = 'train'                              # The split used to training
VAL_SPLIT = 'val'                                  # The split used to evaluation
SNAPSHOT_DIR = './logs/'                           # The directory where the logs are saved
IGNORE_LABEL = 255                                 # Ignore label
NUM_TRAIN_SAMPLES = 1464                           # The number of training set
NUM_VAL_SAMPLES = 1449                             # The number of validation set
RESTORE_FROM = None                                # Path to pretrained model 
PRETRAINED_MODEL = None                            # Pretraind model name that can be "resnet_v2_101" or "vgg_16"
SEGMENTATION_MODEL = "FCN8"                        # Semantic segmentation model  

MIN_SCALE_FACTOR = 0.5                             
MAX_SCALE_FACTOR = 2.0                             
SCALE_FACTOR_STEP_SIZE = 0.25                      
RANDOM_SEED = 1234                             
INPUT_SIZE = '320,320'                             
BATCH_SIZE = 4                                     
LEARNING_RATE = 1e-3                             
MOMENTUM = 0.9                                  
POWER = 0.9   
WEIGHT_DECAY = 0.0005                            

NUM_STEPS = 100001                                 # Number of training steps.
SAVE_NUM_IMAGES = 2                                # How many images to save.
SAVE_PRED_EVERY = 100                              # Save summaries every often.
SAVE_MODEL_EVERY = 5000                           # Save model every often.


def get_arguments():
    """
      Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Semantic Segmentation Network")
    
    # Dataset settings.
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                                    help="Path to the directory containing dataset.")
    parser.add_argument("--dataset", type=str, default=DATA_SET, 
                                    help="The dataset name ")
    parser.add_argument("--train-split", type=str, default=TRAIN_SPLIT,
                                    help="The split used to training")
    parser.add_argument("--val-split", type=str, default=VAL_SPLIT,
                                    help="The split used to val")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                                    help="Where to save snapshots of the model.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                                    help="The index of the label to ignore during the training.")
    parser.add_argument("--num-train-samples", type=int, default=NUM_TRAIN_SAMPLES,
                                    help="Number of samples of dataset to train.")
    parser.add_argument("--num-val-samples", type=int, default=NUM_VAL_SAMPLES,
                                    help="Number of samples of dataset to val.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM, 
                                    help="Where restore model parameters from.")
    parser.add_argument("--pretrained-model", type=str, default=PRETRAINED_MODEL, 
                                    help="Pretraind model name.")  
    parser.add_argument("--segmentation-model", type=str, default=SEGMENTATION_MODEL, 
                                    help="Semantic segmentation model name.")      

    # Settings for data augment.
    parser.add_argument("--min-scale-factor",type=float,default=MIN_SCALE_FACTOR,
                                    help="set min scale factor")
    parser.add_argument("--max-scale-factor",type=float,default=MAX_SCALE_FACTOR,
                                    help="set max scale factor")
    parser.add_argument("--scale-factor-step-size",type=float,default=SCALE_FACTOR_STEP_SIZE,
                                    help="set scale factor step size")                                       
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED, 
                                    help="Random seed to have reproducible results.")

    # Settings for training strategy. 
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                                    help="Comma-separated string with height and width of images.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                                    help="Number of images sent to the network in one step.")
    # For a small batch size, it is better to keep the statistics of the BN layers 
    # (running means and variances) frozen, and to not update the values provided by 
    # the pre-trained model. If is_training=True, the statistics will be updated 
    # during the training. Note that is_training=False still updates BN parameters 
    # gamma (scale) and beta (offset) if they are presented in var_list of the optimiser definition.    
    parser.add_argument("--is-training", action="store_false",
                                    help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                                    help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                                    help="Momentum component of the optimiser.")
    # Note that when the "action" is set to "store_true", then its value is false and vice versa.
    parser.add_argument("--not-restore-last", action="store_true",
                                    help="Whether to not restore last (psp) layers.")    
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS, 
                                    help="Number of training steps.")
    parser.add_argument("--power", type=float, default=POWER, 
                                    help="Decay parameter to compute the learning rate.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                                    help="Regularisation parameter for L2-loss.")
    
    # Settings for logging.
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                                    help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                                    help="Save summaries every often.")
    parser.add_argument("--save-model-every", type=int, default=SAVE_MODEL_EVERY,
                                    help="Save checkpoint every often.")

    return parser.parse_args()

def save(saver, sess, logdir, step):
   '''
     Save weights.
   '''
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)
    
   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    '''
      Load trained weights.   
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def main():
    """
        Create the model and start the training.
    """

    args = get_arguments()

    if (args.dataset not in DATASET_LIST):
        print ("The dataset is not supported")
        return False
    if (args.segmentation_model not in SEGMENTATION_MODEL_LIST):
        print ("The segmentation model is not supported")
        return False
    
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    num_classes = DATASET_TO_CLASSES[args.dataset]
    
    tf.set_random_seed(args.random_seed)    

    print ("load data...")
    
    # Load training batch
    train_samples=seg_dataset.get_sample( 
                                    dataset_name = args.dataset, 
                                    split = args.train_split, 
                                    dataset_dir = args.data_dir,
                                    train_crop_size = input_size,
                                    num_classes = num_classes,
                                    num_samples = args.num_train_samples,
                                    ignore_label = args.ignore_label,
                                    min_scale_factor = args.min_scale_factor,
                                    max_scale_factor = args.max_scale_factor,
                                    scale_factor_step_size = args.scale_factor_step_size,
                                    is_training = True,
                                    batch_size = args.batch_size,
                                    mean_pixel = IMG_MEAN)
    # Load validation batch
    val_samples=seg_dataset.get_sample( 
                                    dataset_name = args.dataset, 
                                    split = args.val_split, 
                                    dataset_dir = args.data_dir,
                                    train_crop_size = input_size,
                                    num_classes = num_classes,
                                    num_samples = args.num_val_samples,
                                    ignore_label = args.ignore_label,                                    
                                    min_scale_factor = 1,
                                    max_scale_factor = 1,
                                    scale_factor_step_size = 0,
                                    is_training = False,
                                    batch_size = args.batch_size,
                                    mean_pixel = IMG_MEAN)
    
    print ("create network...")

    image_batch = tf.placeholder(dtype=tf.float32, shape = [args.batch_size,input_size[0], input_size[1], 3])
    label_batch = tf.placeholder(dtype=tf.uint8, shape = [args.batch_size,input_size[0], input_size[1], 1])    
    output = build_model(inputs = image_batch, num_classes = num_classes, segmentation_model = args.segmentation_model, is_training = args.is_training)

    # define the variables that need to restore from pretraind model
    restore_var = [v for v in tf.global_variables() if args.pretrained_model in v.name]

    # compute cross_entropy as loss 
    pred_vector = tf.reshape(output, [-1, num_classes])
    label_proc = prepare_label(label_batch, output, num_classes=num_classes, one_hot=False) 
    
    label_vector = tf.reshape(label_proc, [-1,])
    indices = tf.squeeze(tf.where(tf.less_equal(label_vector, num_classes - 1)), 1)
    label_vector = tf.cast(tf.gather(label_vector, indices), tf.int32)
    pred_vector = tf.gather(pred_vector, indices)     
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = pred_vector, labels = label_vector)
    l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)
    
    # obtain final segmentation results vis bilinear interpolation
    input_shape = tf.shape(image_batch)[1:3]
    pred_img = tf.image.resize_bilinear(output, size = input_shape)     
    pred_img = tf.argmax(pred_img, axis = 3)    
    pred_img = tf.expand_dims(pred_img, axis = 3)    
    pred_img = tf.cast(pred_img, tf.uint8)
    
    # Add summaries for images, labels, semantic predictions and loss
    images_summary = tf.py_func(inv_preprocess, [image_batch, args.save_num_images, IMG_MEAN], tf.uint8)
    labels_summary = tf.py_func(decode_labels, [label_batch, args.dataset, args.save_num_images], tf.uint8)
    preds_summary = tf.py_func(decode_labels, [pred_img, args.dataset, args.save_num_images], tf.uint8)
    image_summary = tf.summary.image('images', 
                                     tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary]), 
                                     max_outputs=args.save_num_images) # Concatenate row-wise.
    loss_summary = tf.summary.scalar('loss',reduced_loss)
    total_summary = tf.summary.merge_all()
    train_summary_writer = tf.summary.FileWriter(args.snapshot_dir + '/train', graph=tf.get_default_graph())
    val_summary_writer = tf.summary.FileWriter(args.snapshot_dir + '/val', graph=tf.get_default_graph())
   
    # define learning rate strategy
    base_lr = tf.constant(args.learning_rate)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    global_step = tf.get_variable(name="global_step",initializer=0,trainable = False)
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))
    
    # Build the optimizer
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)   
    train_op = tf.train.MomentumOptimizer(learning_rate, args.momentum).minimize(reduced_loss)
        
    # start session and initialize global variables and local variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) 
    sess.run(init)
    
    # create a saver to save or restore model 
    saver = tf.train.Saver(var_list = tf.global_variables(), max_to_keep=5)
    if args.restore_from is not None:
        loader = tf.train.Saver(var_list = restore_var)
        load(loader, sess, args.restore_from)
    
    print ("start training...")    
    
    # Before loading data, we need to create coordinator and start queue runner.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for step in range(args.num_steps):  
        
        print ("global step:", step)  
        train_batch = sess.run(train_samples)        
        feed_dict = { step_ph : step , 
                      image_batch : train_batch['image'],
                      label_batch : train_batch['label'] }
        sess.run([train_op,update_ops], feed_dict = feed_dict)
        
        # save summaries every often.
        if step % args.save_pred_every == 0:
            summary = sess.run(loss_summary, feed_dict = feed_dict)
            train_summary_writer.add_summary(summary, step)
            
            val_batch = sess.run(val_samples)
            feed_dict = { step_ph : step , 
                          image_batch:val_batch['image'], 
                          label_batch : val_batch['label'] }
            summary = sess.run(total_summary, feed_dict = feed_dict)
            val_summary_writer.add_summary(summary, step)
        
        # save checkpoint every often.
        if step % args.save_model_every == 0:
            save(saver, sess, args.snapshot_dir, step)
       
    print ("train over !!!")  

    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()