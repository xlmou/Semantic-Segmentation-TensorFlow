from PIL import Image
import numpy as np
import tensorflow as tf
from collections import namedtuple

# pascal voc2012 colour map including 21 classes
pascal_voc2012_colors = [[0,0,0]
                        # 0=background
                        ,[128,0,0],[0,128,0],[128,128,0],[0,0,128],[128,0,128]
                        # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                        ,[0,128,128],[128,128,128],[64,0,0],[192,0,0],[64,128,0]
                        # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                        ,[92,128,0],[64,0,128],[192,0,128],[64,128,128],[192,128,128]
                        # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                        ,[0,64,0],[128,64,0],[0,192,0],[128,192,0],[0,64,128]]
                        # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

# cityscapes color_map including 19 classes
cityscapes_colors = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], 
                     [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], 
                     [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60], 
                     [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], 
                     [0, 0, 230], [119, 11, 32]]

# ADE20K color_map including 151 classes
ADE20K_colors =[[0,0,0],[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3], 
                [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230], [4, 250, 7], 
                [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51],
                [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3], 
                [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71], 
                [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92], [112, 9, 255], 
                [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71], [255, 41, 10], 
                [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7], 
                [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255], 
                [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140], [250, 10, 15], 
                [20, 255, 0], [31, 255, 0], [255, 31, 0], [255, 224, 0], [153, 255, 0], 
                [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255], 
                [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255], [0, 255, 112], 
                [0, 255, 133], [255, 0, 0], [255, 163, 0], [255, 102, 0], [194, 255, 0], 
                [0, 143, 255], [51, 255, 0], [0, 82, 255], [0, 255, 41], [0, 255, 173], 
                [10, 0, 255], [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255], 
                [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20], [255, 184, 184], 
                [0, 31, 255], [0, 255, 61], [0, 71, 255], [255, 0, 204], [0, 255, 194], 
                [0, 255, 82], [0, 10, 255], [0, 112, 255], [51, 0, 255], [0, 194, 255], 
                [0, 122, 255], [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0], 
                [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0], [8, 184, 170], 
                [133, 0, 255], [0, 255, 92], [184, 0, 255], [255, 0, 31], [0, 184, 255], 
                [0, 214, 255], [255, 0, 112], [92, 255, 0], [0, 224, 255], [112, 224, 255], 
                [70, 184, 160], [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163], 
                [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0], [255, 0, 235], 
                [245, 0, 255], [255, 0, 122], [255, 245, 0], [10, 190, 212], [214, 255, 0], 
                [0, 204, 255], [20, 0, 255], [255, 255, 0], [0, 153, 255], [0, 41, 255], 
                [0, 255, 204], [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255], 
                [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255], [184, 255, 0], 
                [0, 133, 255], [255, 214, 0], [25, 194, 194], [102, 255, 0], [92, 0, 255]]

def decode_labels(mask, dataset, num_images = 1):
    """Decode batch of segmentation masks of pascal voc 2012、cityscapes and ADE20K
    
    Args:
      mask: result of inference after taking argmax.
      dataset : dataset name.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    dataset_name = dataset.decode(encoding='utf-8')
    cmap = []
    if dataset_name=="pascal_voc2012":      
      for elem in pascal_voc2012_colors:
        cmap.extend(elem)
      padding = [0 for x in range(768 - len(cmap))]
      cmap.extend(padding)
    if dataset_name=="cityscapes":
      for elem in cityscapes_colors:
        cmap.extend(elem)
      padding = [0 for x in range(768 - len(cmap))]
      cmap.extend(padding)
    if dataset_name=="ADE20K":
      for elem in ADE20K_colors:
        cmap.extend(elem)
      padding = [0 for x in range(768 - len(cmap))]
      cmap.extend(padding)            

    n, h, w, c = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = np.squeeze(mask[i])
      img = Image.fromarray(img,'P')
      img.putpalette(cmap)
      img = img.convert('RGB')
      outputs[i] = np.array(img)
    return outputs


def prepare_label(input_batch, output_batch, num_classes, one_hot=False):
    """Resize masks and perform one-hot encoding.

    Args:
      input_batch: input tensor of shape [batch_size, H, W, 1].
      output_batch: prediction tensor of shape [batch_size, h', w', num_classes] 
      num_classes: number of classes to predict (including background).
      one_hot: whether perform one-hot encoding.

    Returns:
      A tensor of shape [batch_size, h', w', 1]

    """
    with tf.name_scope('label_encode'):
        output_shape = tf.shape(output_batch)[1:3]
        input_shape = tf.shape(input_batch)[1:3]
        resized_input = tf.image.resize_nearest_neighbor(input_batch, output_shape)
        output = tf.cond(tf.logical_and(tf.equal(input_shape[0], output_shape[0]), 
                                        tf.equal(input_shape[1], output_shape[1])), 
                          lambda:input_batch, lambda:resized_input)
        # reducing the channel dimension.
        output = tf.squeeze(output, axis = [3]) 
        if one_hot:
            output = tf.one_hot(output, depth=num_classes)
    return output

def inv_preprocess(imgs, num_images, img_mean):
    """Inverse preprocessing of the batch of images.
       Add the mean vector
       
    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.
  
    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    n, h, w, c = imgs.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, c), dtype=np.uint8)
    for i in range(num_images):
        outputs[i] = (imgs[i]*255 + img_mean).astype(np.uint8)
    return outputs  


if __name__ == '__main__':
  img = Image.open('D:/test/003.png')
  img = np.array(img)
  img = np.expand_dims(img, axis=0)
  img = np.expand_dims(img, axis=3)
  img = decode_labels(img,dataset='ADE20K',num_images = 1)
  color=Image.fromarray(img[0],'RGB')
  color.show()
