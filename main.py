import numpy as np
from imutils import face_utils
import cv2
import dlib


# Shape Detector
shape_detector = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

# face descriptor
shape_descriptor = dlib.face_recognition_model_v1('./models/dlib_face_recognition_resnet_model_v1.dat')

img = cv2.imread('./images/face.jpg')

image = img.copy()

face_detector = dlib.get_frontal_face_detector()
faces = face_detector(image)

for box in faces:
    pt1 = box.left(),box.top()
    pt2 = box.right(),box.bottom()

    face_shape = shape_detector(image,box)
    face_shape_array = face_utils.shape_to_np(face_shape,"int")
    # face_descriptor = shape_detector.compute_face_descriptor(image,face_shape)

    for point in face_shape_array:
        cv2.circle(image,tuple(point),3,(0,255,0),-1)
        cv2.rectangle(image,pt1,pt2,(0,255,0),3)

cv2.imshow('Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()