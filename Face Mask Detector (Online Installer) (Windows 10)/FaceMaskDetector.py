# importing necessary packages
from __future__ import print_function

import os
#for reducing unnecessary issue to those Devices which doesn't has GPU support
#hiding presence of GPU devices
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import winsound
from PIL import Image
from PIL import ImageTk
import datetime
from win10toast import ToastNotifier


def detectAndPredictMask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations, and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI (Region Of Interest), convert it from BGR to RGB channel ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all* faces at the same time rather than one-by-one predictions in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding locations
	return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = r"deploy.prototxt"
weightsPath = r"res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# display for Exit option
# create an object to ToastNotifier class
n = ToastNotifier()
n.show_toast("Face Mask Detector", "Press 'q' to Quit Application", duration = 1.5)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it to have a maximum width of Max System Width 1000
	frame = vs.read()
	frame = imutils.resize(frame, width=1000)

	# detect faces in the frame and determine if they are wearing a face mask or not
	(locs, preds) = detectAndPredictMask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# display the label and bounding box rectangle on the output frame
		# in few devices winsound.Beep was not working properly and raising errors, to avoid that try-except is used
		try:
			if label=="No Mask": winsound.Beep(2000,50)
		except :
			pass

		cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv2.imshow("Face Mask Detector     : Press 'q' to Quit Application", frame)

	#cv2.waitKey(1) waits for up to 1 ms for a keypress and then checks if 'q' was pressed or not
	key = cv2.waitKey(1) & 0xFF
    
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
