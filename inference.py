import os
import sys
import time
import argparse
from typing import Union, List, Tuple

import cv2
import numpy as np
import imutils
from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

from model import generate_model


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--classifier", type=str, required=True,
	help="path to the classifier model")
ap.add_argument("-d", "--detector", type=str, required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-p", "--path", type=str, required=True, default=0,
	help="path to the input file(s), can be image, video or a camera ID.")
ap.add_argument("-v", "--video", action='store_true',
	help="detect video type as the input")
ap.add_argument("-i", "--image", action='store_true',
	help="detect image type as the input")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum threshold for prediction probability to filter weak detections")


class SpoofRecog:
	def __init__(self, detector, classifier, confidence):
		"""
		Init
		"""
		self.detector = detector
		self.classifier = classifier
		self.confidence = confidence
		self.label = ["Real", "Spoof"]

	def image(self, imgsrc: str) -> None:
		"""
		Face anti-spoofing detection on image(s) input.
		:param
			vidsrc: Video source (use camera ID or the video filepath).
		"""
		frame = cv2.imread(imgsrc)
		frame = self._process_frame(frame)
		cv2.imshow("Frame", frame)
		cv2.waitKey(1)

	def video(self, vidsrc: Union[int, str] = 0) -> None:
		"""
		Face anti-spoofing detection on video input.
		:param
			vidsrc: Video source (use camera ID or the video filepath).
		"""
		# initialize the video stream and allow the camera sensor to warmup
		print("[INFO] starting video stream...")
		vs = VideoStream(src=vidsrc).start()
		time.sleep(2.0)

		# loop over the frames from the video stream
		while True:
			# grab the frame from the threaded video stream and resize it
			# to have a maximum width of 600 pixels
			frame = vs.read()
			frame = self._process_frame(frame)

			# show the output frame and wait for a key press
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF
			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break

		# do a bit of cleanup
		cv2.destroyAllWindows()
		vs.stop()

	def _process_frame(self, frame: np.array) -> np.array:
		"""
		Perform face detection and spoof classification on a frame
		:param
			frame: Input frame to detect and classify.
		:return
			frame: Processed frame; Adding bounding box and the label prediction(s).
		"""
		frame = imutils.resize(frame, width=600)

		# grab the frame dimensions and convert it to a blob
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
									 scalefactor=1.0,
									 size=(300, 300),
									 mean=(104.0, 177.0, 123.0)
									 )
		# pass the blob through the network and obtain the detections and
		# predictions
		self.detector.setInput(blob)
		detections = self.detector.forward()

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with the
			# prediction
			confidence = detections[0, 0, i, 2]
			# filter out weak detections
			if confidence > self.confidence:
				# compute the (x, y)-coordinates of the bounding box for
				# the face and extract the face ROI
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				# ensure the detected bounding box does fall outside the
				# dimensions of the frame
				startX = max(0, startX)
				startY = max(0, startY)
				endX = min(w, endX)
				endY = min(h, endY)
				# extract the face ROI and then preproces it in the exact
				# same manner as our training data
				face = frame[startY:endY, startX:endX]
				face = cv2.resize(face, (96, 96))
				face = face.astype("float") / 255.0
				face = img_to_array(face)
				face = np.expand_dims(face, axis=0)
				# pass the face ROI through the trained liveness detector
				# model to determine if the face is "real" or "fake"
				preds = float(self.classifier.predict(face)[0])
				j = 0 if preds < 0.5 else 1
				label = self.label[j]

				# draw the label and bounding box on the frame
				label = "{}".format(label)
				# label = "{}: {:.4f}".format(label, preds[j])
				cv2.putText(frame, label, (startX, startY - 10),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY),
							  (0, 0, 255), 2)
		return frame


if __name__ == "__main__":
	# ---------------------------  DEBUGGING SECTION  ---------------------------
	debug_input = ["inference.py",
				   "--classifier", "./output/lcc-train01-weight/mobilenetv2-best.hdf5",
				   "--detector", "./pretrain/detector",
				   "--path", "0",  # "./input/demo/20777.jpg",
				   "--video",  # "--image",
				   "--confidence", "0.5"]
	sys.argv = debug_input  # Uncomment for DEBUGGING purpose!

	# -------------------------------  START HERE  -------------------------------
	args = vars(ap.parse_args())  # Initialize the input argument(s)

	# Load the Face detection model
	print("[INFO] loading face detector...")
	protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
	modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
	detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

	# Load the face spoofing classifier
	print("[INFO] loading liveness detector...")
	# classifier = load_model(args["classifier"])
	classifier = generate_model(args["classifier"], shape=(96, 96))

	# Threshold for face detection
	confidence_threshold = args["confidence"]

	# Main process
	pathsrc = args["path"]
	model = SpoofRecog(detector, classifier, confidence_threshold)  # Instantiate the model

	if args["video"]:  # Detect on video input
		try:
			video_source = int(pathsrc)
		except:
			video_source = pathsrc
		print(f"Processing VIDEO input from '{pathsrc}'...")
		model.video(vidsrc=video_source)

	elif args["image"]:  # Detect on image input(s)
		print(f"Processing IMAGE input from '{pathsrc}'...")
		model.image(imgsrc=pathsrc)

	else:
		raise ValueError(f"Define the input type! Choose between --image or --video")

	print("Finished!")
