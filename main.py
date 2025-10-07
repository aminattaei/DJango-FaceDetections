import numpy as np
import pandas as pd
import cv2
import os
import pickle


face_detection_model = './models/res10_300x300_ssd_iter_140000_fp16.caffemodel'
face_detection_prot = './models/deploy.prototxt.txt'
face_descriptor = './models/openface.nn4.small2.v1.t7'

# load models
detector_model = cv2.dnn.readNetFromCaffe(face_detection_prot, face_detection_model)
descriptor_model = cv2.dnn.readNetFromTorch(face_descriptor)


img = cv2.imread('./images/face.jpg')

image = img.copy()
h,w = image.shape[:2]

img_blob = cv2.dnn.blobFromImage(image,1,(300,300),(104,177,123),swapRB=False,crop=False)
detector_model.setInput(img_blob)

detections = detector_model.forward()

def helper(image_path):
    img = cv2.imread(image_path)
    # step-1: face detection
    image = img.copy()
    h,w = image.shape[:2]
    img_blob = cv2.dnn.blobFromImage(image,1,(300,300),(104,177,123),swapRB=False,crop=False)
    # set the input
    detector_model.setInput(img_blob)
    detections = detector_model.forward()

    if len(detections) > 0:
        i = np.argmax(detections[0,0,:,2])
        confidence = detections[0,0,i,2]
        if confidence > 0.5:
            box = detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startx,starty,endx,endy) = box.astype('int')
            # step-2: Feature Extraction or Embedding
            roi = image[starty:endy,startx:endx].copy()
            # get the face descriptors
            faceblob = cv2.dnn.blobFromImage(roi,1/255,(96,96),(0,0,0),swapRB=True,crop=True)
            descriptor_model.setInput(faceblob)
            vectors = descriptor_model.forward()
            
            return vectors
    return None


data = dict(data=[], label=[])

folders = os.listdir('people-images')
for folder in folders:
    filenames = os.listdir('people-images/{}'.format(folder))
    for filename in filenames:
        try:
            vector = helper('./people-images/{}/{}'.format(folder, filename))
            if vector is not None:
                data['data'].append(vector)
                data['label'].append(folder)
        except:
            print('error')

for i in range(0,detections.shape[2]):
    confidence = detections[0,0,i,2]

    if confidence > 0.5:
        box = detections[0,0,i,3:7] * [w,h,w,h]
        (x1,y1,x2,y2) = box.astype("int")
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)

pickle.dump(data,open('data_face_features.pickle', mode='wb'))


cv2.imshow('Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()