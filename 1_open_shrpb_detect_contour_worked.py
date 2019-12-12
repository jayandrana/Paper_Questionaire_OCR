import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import pickle

def predict(img_cropped1):
	(h1,w1)= img_cropped1.shape[:2]
	#im_resized = cv2.resize(img_cropped1,(20,20), interpolation=cv2.INTER_LINEAR)
	img_gray = cv2.cvtColor(img_cropped1, cv2.COLOR_BGR2GRAY)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 2)).astype("uint8")
	thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)


	print("gray")
	print(thresh)
	plt.imshow(thresh)
	plt.show()
	##Another Contour Detection in Action
	##Recognize Border Thickness
	#1st layer
	for y in range(0,h1):
		not_black_pixel = 0
		for x in range(0,w1):
			if(thresh[y,x]>0):
				not_black_pixel = not_black_pixel + 1
		border_threshold =  not_black_pixel/w1
		#print(" Border Treshold : "+str(y)+" - "+str(border_threshold))
		if (border_threshold > 0.7):
			#print("Border Found : "+str(y))
			for x in range(0,w1):
				thresh[y,x] = 0

	for x in range(0,w1):
		not_black_pixel = 0
		for y in range(0,h1):
			if(thresh[y,x]>0):
				not_black_pixel = not_black_pixel + 1
		border_threshold =  not_black_pixel/w1
		print(" Border Treshold : "+str(x)+" - "+str(border_threshold))
		if (border_threshold > 0.65):
			print("Border Found : "+str(y))
			for y in range(0,h1):
				thresh[y,x] = 0


	cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:1000]
	upy = 0
	leftx = 0
	my_w = 0
	my_h =0
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.019 * peri, True)
		x, y, w, h = cv2.boundingRect(approx)
		print(x, y, w, h)
		if (h/h1 > 0.5):
			upy=y
			leftx = x
			my_w = w
			my_h = h

	img_cropped2 = thresh[upy:upy+my_h, leftx:leftx+my_w]

	print(img_cropped2)
	plt.imshow(img_cropped2)
	plt.show()

	## Alternative algorithm
	#1. Get random coordinate somewhere in the middle (define middle area)
	# middle area is 50% middle area, its a range

	middle_area_x_min = w1/4
	middle_area_x_len = w1/2

	middle_area_y_min = h1/4
	middle_area_y_len = h1/2

	# For LOOP
	#2. if the coordinate is nonzero then it's a seed of a cluster
	#3. check y+1, y-1, x+1, x-1 pixel sourounding the seed
	#4. if nonzero and not exist in the cluster then add to cluster
	#5. flag the seed with checked sign
	#6. proceed to the next pixel in the cluster thet are not checked as seed
	#WHILE THERE'S STILL pixel there are not flagged checked

	#reSize to 20x20
	im_resized = cv2.resize(img_cropped2,(20,20))
	im_resized = im_resized/255
	print(im_resized)
	plt.imshow(im_resized)
	plt.show()

	#Create a Zero NP Array 2D
	new28x28 = np.zeros([28,28])
	#put 20x20 image to this canvas
	i=0
	j=0
	for x in range(4,24):
	    i = x - 4
	    for y in range(4,24):
	        j = y - 4
	        new28x28[y,x] = im_resized[j,i]


	print(new28x28)
	plt.imshow(new28x28)
	plt.show()





	plt.show()
	filename = 'finalized_model_svm_10_Nov.sav'
	loaded_model = pickle.load(open(filename, 'rb'))
	result = loaded_model.predict(new28x28.reshape(1,-1))
	print(result)



filename1 = "shprB_1_1 001.jpg"
im = cv2.imread(filename1)

gray =  cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 30,200)

(h,w)= edged.shape[:2]
cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:1000]

# loop over our contours
arr_contour = []
i=0
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.01 * peri, True)
	x, y, w, h = cv2.boundingRect(approx)
	my_box_contour = []
	if w > 70 and w < 80:
		#print (w)
		screenCnt = approx
		my_box_contour.append(x)
		my_box_contour.append(y)
		my_box_contour.append(w)
		my_box_contour.append(h)
		#first contours
		if(i==0):
			arr_contour.append(my_box_contour)
		i=i+1
		#cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 1)
		#img_cropped1 = im[y:y+h, x:x+w]
		#plt.show()
		cnt_exist = 0
		for cnt in arr_contour:
			if(cnt[1]-y < 2 ):
				print ("y : "+str(y))
				print ("cnt[1] :"+str(cnt[1]))
				cnt_exist =1
				break;
		if(cnt_exist == 0):
			arr_contour.append(my_box_contour)

#remove contour yang berhimpitan
print (arr_contour)
choosen=[]
for i in range (len(arr_contour)):
	print (arr_contour[i])
	img_cropped1 = im[arr_contour[i][1]:arr_contour[i][1]+arr_contour[i][3], arr_contour[i][0]:arr_contour[i][0]+arr_contour[i][2]]

	plt.imshow(img_cropped1)
	plt.show()
	predict(img_cropped1)



	#peri = cv2.arcLength(c, True)
	#approx = cv2.approxPolyDP(c, 0.01 * peri, True)

	# if our approximated contour has four points, then
	# we can assume that we have found our screen
	#if len(approx) == 4:
		#screenCnt = approx
		#break


#cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 1)
#img_cropped1 = im[y:y+h, x:x+w]
#plt.imshow(img_cropped1)
#plt.show()
#print (h,w)
## Original Document SIze 3507x2550
## coba gambar MRZ secara manual
#x = 774
#y = 597
#h =  45
#w = 61
#cv2.rectangle(edged, (x, y), (x + w, y + h), (0, 255, 0), 1)
#636 ,  594
#703 ,  595
#772 ,  594
#838 ,  594
#856 ,  592
#639 ,  643
#705 ,  643
#773 ,  643
#838 ,  643
#img_cropped1 = edged[y:y+h, x:x+w]


#print("gray")
#print(img_cropped1)
#plt.imshow(img_cropped1)

#plt.show()
