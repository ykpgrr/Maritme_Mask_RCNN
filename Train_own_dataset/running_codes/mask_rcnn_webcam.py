import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
sys.path.insert(0,'/home/adllo/others_git/Mask_RCNN/mrcnn')  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import Maritime config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import maritime
 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
#MARITIME_MODEL_PATH = os.path.join("/home/adllo/others_git/Mask_RCNN/mrcnn/logs/100_images/mask_rcnn_maritime_0160.h5")
MARITIME_MODEL_PATH = os.path.join("/home/adllo/others_git/Mask_RCNN/mrcnn/saved_weights/100_images/mask_rcnn_maritime_0160.h5")

class InferenceConfig(maritime.MaritimeConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MARITIME dataset
model.load_weights(MARITIME_MODEL_PATH, by_name=True)

# MARITIME Class names
class_names = ['undetected','buoy','land','sea','ship','sky']
colors= [(0,0,0),(0,1,0),(0,0,0.5),(1,0,0),(1,1,1),(1,1,0)]


def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    # If there are no lines to draw, exit.
    if lines is None:
        return
    # Make a copy of the original image.
    img = np.copy(img)
    # Create a blank image that matches the original in size.
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_image, 1.0, 0.0)
    # Return the modified image.
    return img

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c]*255,
                                  image[:, :, c])
    #cv2.imshow("TEST",image.astype(np.uint8))
    #print(color)
    return image


def create_masked_image(image, masks, class_ids):

	global mask_sky
	mask_sky_bool = False

	# Generate random colors
	N = masks.shape[-1]
	#print("Number of instances: " + str(N))
	#if (N == 0): 
	#	return image.astype(np.uint8)

	# Show area outside image boundaries.
	height, width = image.shape[:2]
	masked_image = image.astype(np.uint32).copy()

	for i in range(N):
	
		#Label
		class_id = class_ids[i]
		label = class_names[class_id]
		#print(label)

		global colors
		color = colors[class_id]
		#print(color)
		'''		
		#Save sky mask
		if (label=='sky'):
			#mask_sky  = cv2.cvtColor(masks[:, :, i].dtype='uint8', cv2.COLOR_BGR2GRAY)
			mask_sky  = masks[:, :, i].dtype='uint8'
			print(mask_sky)
			mask_sky_bool = True
		'''		
		# Mask
		mask = masks[:, :, i]
		masked_image = apply_mask(masked_image, mask, color, 0.5)
		#cv2.imshow(str(label),masked_image.astype(np.uint8))
	'''
	if (mask_sky_bool):
		lines = cv2.HoughLinesP(
			mask_sky,
			rho=6,
			theta=np.pi / 60,
			threshold=160,
			lines=np.array([]),
			minLineLength=40,
			maxLineGap=25
		)
		masked_image = draw_lines(masked_image, lines)
	'''

	return masked_image.astype(np.uint8)


if __name__ == '__main__':

	# Start processing images from webcam
	cap = cv2.VideoCapture(0)

	while(True):

		
		#start=time.time()
		### Capture frame-by-frame
		ret, frame = cap.read()
		#fra = time.time()
		#print ('Grab frame', str(fra-start))

		### Run detection
		results = model.detect([frame], verbose=0)
		#res = time.time()
		#print ('Detection', str(res-fra))

		### Visualize results
		r = results[0]
		#print(r['class_ids'])
		masked_image = create_masked_image(frame, r['masks'], r['class_ids'])
		#vis = time.time()
		#print ('Visualization', str(vis-res))

		# Display the resulting frame
		cv2.imshow('Mask_rcnn results',masked_image)
		if cv2.waitKey(1) & 0xFF == ord('q'):
		    break
		
	   


	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()





