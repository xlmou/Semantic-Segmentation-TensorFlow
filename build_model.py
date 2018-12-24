#coding=utf-8

import tensorflow as tf
slim = tf.contrib.slim

#------------High-quality segmentation network-------------
from model.FCN8 import FCN8
from model.U_Net import U_Net
from model.Seg_Net import Seg_Net
from model.Deeplab_v1 import Deeplab_v1
from model.Deeplab_v2 import Deeplab_v2
from model.Deeplab_v3 import Deeplab_v3
from model.PSPNet import PSPNet
from model.GCN import GCN

#------------Real-time segmentation network----------------
from model.ENet import ENet
from model.ICNet import ICNet 

def build_model(inputs , num_classes, segmentation_model, is_training ):
	if segmentation_model=="FCN8":
		print ("segmentation_model:FCN8")
		return FCN8(inputs , num_classes)

	elif segmentation_model=="U_Net":
		print ("segmentation_model:U_Net")
		return U_Net(inputs , num_classes)

	elif segmentation_model=="Seg_Net":
		print ("segmentation_model:Seg_Net")
		return Seg_Net(inputs , num_classes)

	elif segmentation_model=="Deeplab_v1":
		print ("segmentation_model:Deeplab_v1")
		return Deeplab_v1(inputs , num_classes)

	elif segmentation_model=="Deeplab_v2":
		print ("segmentation_model:Deeplab_v2")
		return Deeplab_v2(inputs , num_classes, is_training)

	elif segmentation_model=="Deeplab_v3":
		print ("segmentation_model:Deeplab_v3")
		return Deeplab_v3(inputs , num_classes, is_training)

	elif segmentation_model=="PSPNet":
		print ("segmentation_model:PSPNet")
		return PSPNet(inputs , num_classes, is_training)

	elif segmentation_model=="GCN":
		print ("segmentation_model:GCN")
		return GCN(inputs, num_classes, is_training)

	elif segmentation_model=="ENet":
		print ("segmentation_model:ENet")
		return ENet(inputs, num_classes, is_training)

	elif segmentation_model=="ICNet":
		print ("segmentation_model:ICNet")
		return ICNet(inputs , num_classes, is_training)
	
# test
if __name__ == '__main__':
	inputs = tf.placeholder(dtype = tf.float32, shape = [4, 224, 224, 3])
	logits = build_model(inputs = inputs, segmentation_model='ICNet', num_classes = 21, is_training = True)
	print (logits)
