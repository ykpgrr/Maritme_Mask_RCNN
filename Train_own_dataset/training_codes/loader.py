

import os 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import JSON_parser

categories_to_save = ['buoy', 'land', 'sea', 'ship', 'sky']

dataset_path = "/home/adllo/others_git/Mask_RCNN/Train_own_dataset/maritime_dataset/"
image_list_path = "image_list.txt"
image_list_path = os.path.join(dataset_path, image_list_path)

save_dir = dataset_path + "training"


#Load filepaths of images
with open(image_list_path) as f:
    image_list = f.readlines()
# Remove whitespaces
image_list = [x.strip() for x in image_list] 

examples_with_cat = np.full(len(categories_to_save), 0)
total_cat = np.full(len(categories_to_save), 0)
unknown_masks = 0
fillColor = 1

#Loop through all of the examples
for example in range(0, len(image_list)):

	#Get the relative path to the example (strap filename)
	rel_example_path = str(image_list[example])
	#print("rel_example_path: "+str(rel_example_path))
	#Get absolute example image path
	abs_example_path = os.path.join(dataset_path + "images/", rel_example_path)
	#print("abs_example_path: "+str(abs_example_path))
	#Get saving path for this example
	example_save_dir = os.path.join(save_dir, '{0:06}'.format(example))
	#print("example_save_dir: "+str(example_save_dir))

	# Print total images
	print ("Image name: " + str(rel_example_path) + "(" + str(example) + "/" + str(len(image_list)) + ")")


	#Load the frame
	frameData = JSON_parser.readFrame(abs_example_path, categories_to_save)

	if(len(frameData.labels2D) == 0):
		continue

	#Get the masks save dir, create folders if they dont exist
	masks_save_dir = os.path.join(example_save_dir, 'labels')

	cat_saved = np.full(len(categories_to_save), 0)
	cat_present = np.full(len(categories_to_save), False)
	mask_type = "unknown"


	#Loop through all annotations/polygons
	for j in range(0, len(frameData.annotation2D)):
		#print(frameData.annotation2D[j])
		#Check for the vector error
		if frameData.annotation2D[j].size < 3:
			print("DEBUG")
			continue
		
		#Create an empty image to add mask polygon
		mask = np.zeros((frameData.imgRGB.shape[0], frameData.imgRGB.shape[1], 1), dtype=np.uint8)
		#Add polygon to the image
		cv2.fillPoly(mask, np.int32([frameData.annotation2D[j]]), fillColor)


		#Check the label
		for cat_idx, cat in enumerate(categories_to_save):

			if frameData.labels2D[j].lower().find(cat) != -1:
				

				if (frameData.labels2D[j].lower() != cat):
					unknown_masks = unknown_masks + 1
					#print(frameData.labels2D[j])
				
				
				cat_present[cat_idx] = True
				total_cat[cat_idx] = total_cat[cat_idx] + 1 

				#Create a mask name
				mask_name = cat+'_'+'{0:03}'.format(cat_saved[cat_idx])+'.png'

				cat_saved[cat_idx] = cat_saved[cat_idx] + 1

				#Create the folder here
				if not os.path.exists(masks_save_dir):
					os.makedirs(masks_save_dir)

				#Save the image
				cv2.imwrite(os.path.join(masks_save_dir, mask_name), mask)
				
				break

	#Save image only if any label was already saved
	if np.any(cat_present):
		#cv2.imwrite(os.path.join(example_save_dir, '{0:06}'.format(example)+'.jpg'), frameData.imgRGB)
		#cv2.imwrite(os.path.join(example_save_dir, '{0:06}'.format(example)+'_depth.jpg'), frameData.imgRGB)
		cv2.imwrite(os.path.join(example_save_dir, 'rgb.jpg'), frameData.imgRGB)
		

	for idx, present in enumerate(cat_present):
		if present:
			examples_with_cat[idx] = examples_with_cat[idx] + 1		
	
	
for cat_idx, cat in enumerate(categories_to_save):
	print("Files with ", cat, ": ", examples_with_cat[cat_idx])

for cat_idx, cat in enumerate(categories_to_save):
	print("Total instances of ", cat, ": ", total_cat[cat_idx])


print("Unknown masks: ", unknown_masks)
print("FINISHED MASKING IMAGES")
