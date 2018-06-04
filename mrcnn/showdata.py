import maritime
import matplotlib.pyplot as plt
import numpy as np
import utils
import argparse

maskColor = 1

parser = argparse.ArgumentParser(description='Show image with masks')
parser.add_argument("image",metavar='N',type = int,help="Image idx")
args = parser.parse_args()

dataset_dir = "/home/adllo/others_git/Mask_RCNN/Train_own_dataset/maritime_dataset/"
dataset_type = "training"

#Which image to load
image_idx = args.image
print("Loading image: ", image_idx)

#Load config
config = maritime.MaritimeConfig()
#config.display()

#Prepare the dataset loader
dataset_train = maritime.MaritimeDataset()
dataset_train.load_maritime(dataset_dir, dataset_type)
dataset_train.prepare()

#Load data
image = dataset_train.load_image(image_idx)
mask,ids = dataset_train.load_mask(image_idx)

#Split into separate masks
split_masks = np.split(mask, mask.shape[2], 2)

#Combine masks into a single
all_masks = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
all_masks = all_masks * 255

for mask_img in split_masks:
	mask_img = np.reshape(mask_img, [mask_img.shape[0], mask_img.shape[1]])
	all_masks[mask_img == maskColor] = np.random.randint(50, dtype=np.uint8)

#Show original image
plt.imshow(image)

#Show masks
plt.figure(2)
#plt.imshow(all_masks, cmap='gray')
plt.imshow(all_masks)

#Show image with masks
plt.figure(3)
plt.imshow(image)
plt.imshow(all_masks, alpha = 0.3)

plt.show()
