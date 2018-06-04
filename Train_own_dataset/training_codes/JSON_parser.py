import cv2
from os import listdir
import json
import numpy as np
import matplotlib.pyplot as plt

class FrameData:
	def __init__(self, imgRGB, annotation2D,labels2D):
		self.imgRGB = imgRGB
		self.annotation2D = annotation2D
		self.labels2D = labels2D
		
def readFrame(imgPath, categories_to_save):
	#read RGB information to numpy array	
	#rgbPath = "/images/" + img_name + ".JPG"
	#print ("rgbPATH: " + str(imgPath))
	imgRGB = plt.imread(imgPath);

	#read 2D annotations to a list of numpy arrays where each index is related with one object polygon and a list where the index links the object polygon to the object label.
	anotation2D = imgPath.split(".")[0] + ".json"
	#print ("anotation2D: " + str(anotation2D))
	
	with open(anotation2D) as data_file:    
		data = json.load(data_file)
			
	numberOfAnot = len(data["shapes"])
	
	anootation2D = []
	labels2D = []

	for i in range(0,numberOfAnot):
		
		#idxObj = data["shapes"][i];
		#print(idxObj)

		#if idxObj>len(data['shapes']):
		#	continue

		for category in categories_to_save:

			if data['shapes'][i]["label"].lower().find(category) != -1:
				points = []
				x_v = []
				y_v = []
				for l in range(0,len(data["shapes"][i]["points"])):
					x = data["shapes"][i]["points"][l][0]
					y = data["shapes"][i]["points"][l][1]

					x_v.append(x)
					y_v.append(y)
					#print(x_v)
					#print(y_v)
					
				pts2 = np.array([x_v,y_v], np.int32)
				pts2 = np.transpose(pts2);
				anootation2D.append(pts2);
				labels2D.append(data['shapes'][i]["label"])

				break
		
	frameData = FrameData(imgRGB,anootation2D,labels2D)

	return frameData;
