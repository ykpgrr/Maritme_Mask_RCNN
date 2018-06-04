#!/usr/bin/env python

import roslib
roslib.load_manifest('mask_rcnn')
import rospy
import ros_numpy as rnp

import os
import sys
sys.path.append('/home/adllo/others_git/Mask_RCNN/')
sys.path.append('/home/adllo/others_git/coco/PythonAPI') #For pycocotools import
sys.path.append('/home/adllo/others_git/Mask_RCNN/samples/coco/') #For coco import
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import time

import coco
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

#import utils
#import model as modellib
#import visualize

import tensorflow as tf

#%matplotlib inline

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image




#ROOT_DIR = os.path.dirname(os.getcwd())
ROOT_DIR = "/home/adllo/others_git/Mask_RCNN/"

print ("ROOT_DIR: ", ROOT_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mrcnn/saved_weights/mask_rcnn_coco.h5")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
print ("Loading model ............")
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
print ("Loading weights ............")
model.load_weights(COCO_MODEL_PATH, by_name=True)
graph = tf.get_default_graph()

# Load class names
print ("Load classes ...............")
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

colors = visualize.random_colors(len(class_names))
print ("Number of classes", len(class_names))

def create_masked_image(image, masks, class_ids):

	# Generate random colors
	N = masks.shape[-1]
	
	# Show area outside image boundaries.
	height, width = image.shape[:2]
	masked_image = image.astype(np.uint32).copy()

	for i in range(N):
		#Label
		class_id = class_ids[i]
		label = class_names[class_id]

		global colors
		color = colors[i]
		# Mask
		mask = masks[:, :, i]
		masked_image = visualize.apply_mask(masked_image, mask, color)

	return masked_image.astype(np.uint8)



def image_callback(image):
	arr = rnp.numpify(image)

	t0 = time.time()
	# Run detection
	global graph
	with graph.as_default():
		results = model.detect([arr], verbose=1)
	
	# Visualize results
	t1 = time.time()
	r = results[0]
	masked_image = create_masked_image(arr, r['masks'], r['class_ids'])

	print ("Time rcnn: ", t1-t0)
	print ("Time visualization: ", time.time()-t1)

	my_image = rnp.msgify(Image, masked_image, encoding='bgr8')
	img_pub.publish(my_image)
   

if __name__ == '__main__':
	rospy.init_node('mask_rcnn_node', anonymous=True)

	rospy.Subscriber("/camera/rgb/image_rect_color", Image, image_callback, queue_size = 1, buff_size=2**24)
	img_pub = rospy.Publisher('/my_image', Image, queue_size=1)

	rospy.spin()

