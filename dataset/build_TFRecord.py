#codind=utf-8

"""Converts dataset to TFRecord file format with Example protos."""

import math
import os
import random
import sys
import build_data
import tensorflow as tf
import argparse


TRAIN_IMAGE_PATH = './images/train'           # Path to the directory containing train set image.
TRAIN_LABEL_PATH = './profiles/train'         # Path to the directory containing train set label.
VAL_IMAGE_PATH = './images/val'               # Path to the directory containing val set image.
VAL_LABEL_PATH = './profiles/val'             # Path to the directory containing val set label.
OUTPUT_DIR = './tfrecord'                     # Path to directory saving tfrecord files

_NUM_SHARDS = 4                               # The number of tfrecord file per dataset split


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Convert raw images to tfrecord file")
    parser.add_argument("--train-image-path", type=str, default=TRAIN_IMAGE_PATH, 
                                    help="Path to the directory containing train set image")
    parser.add_argument("--train-label-path", type=str, default=TRAIN_LABEL_PATH, 
                                    help="Path to the directory containing train set label")
    parser.add_argument("--val-image-path", type=str, default=VAL_IMAGE_PATH, 
                                    help="Path to the directory containing val set image")
    parser.add_argument("--val-label-path", type=str, default=VAL_LABEL_PATH, 
                                    help="Path to the directory containing val set label")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, 
                                    help="Path to directory saving tfrecord files")
    return parser.parse_args()


def _convert_dataset(dataset_split, dataset_image_dir, dataset_label_dir):
  """Converts the ADE20k dataset into into tfrecord format.

  Args:
    dataset_split: Dataset split (e.g., train, val).
    dataset_image_dir: Dir in which the images locates.
    dataset_label_dir: Dir in which the annotations locates.

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """

  img_names = tf.gfile.Glob(os.path.join(dataset_image_dir, '*.jpg'))
  random.shuffle(img_names)
  seg_names = []
  for f in img_names:
    # get the filename without the extension
    basename = os.path.basename(f).split('.')[0]
    # cover its corresponding *_seg.png
    seg = os.path.join(dataset_label_dir, basename+'.png')
    seg_names.append(seg)

  num_images = len(img_names)
  num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

  image_reader = build_data.ImageReader('jpg', channels=3)
  label_reader = build_data.ImageReader('png', channels=1)

  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(args.output_dir,'%s-%05d-of-%05d.tfrecord' % (dataset_split, shard_id, _NUM_SHARDS))
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, num_images, shard_id))
        sys.stdout.flush()
        
        # read images
        image_filename = img_names[i]
        image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
        height, width = image_reader.read_image_dims(image_data)
        
        # read labels
        seg_filename = seg_names[i]
        seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()
        seg_height, seg_width = label_reader.read_image_dims(seg_data)
        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')
        # Convert to tf example.
        example = build_data.image_seg_to_tfexample(
            image_data, img_names[i], height, width, seg_data)
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


def main(unused_argv):
  args = get_arguments() 
  print (args)
  if not os.path.exists(args.output_dir):
      os.makedirs(args.output_dir) 

  _convert_dataset('train', args.train_image_path, args.train_label_path)
  _convert_dataset('val', args.val_image_path, args.val_label_path)


if __name__ == '__main__':
  tf.app.run()