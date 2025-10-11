import numpy as np
import pandas as pd
import cv2
import os
import pickle

# مدل‌ها
face_detection_model = './models/res10_300x300_ssd_iter_140000_fp16.caffemodel'
face_detection_prot = './models/deploy.prototxt.txt'
face_descriptor = './models/openface.nn4.small2.v1.t7'

# بارگذاری مدل‌ها
detector_model = cv2.dnn.readNetFromCaffe(face_detection_prot, face_detection_model)
descriptor_model = cv2.dnn.readNetFromTorch(face_descriptor)


def extract_face_vector(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    image = img.copy()
    h, w = image.shape[:2]

    # تشخیص چهره
    blob = cv2.dnn.blobFromImage(image, 1, (300, 300), (104, 177, 123), swapRB=False, crop=False)
    detector_model.setInput(blob)
    detections = detector_model.forward()

    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype('int')

            roi = image[y1:y2, x1:x2].copy()
            if roi.size == 0:
                return None

            face_blob = cv2.dnn.blobFromImage(roi, 1 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=True)
            descriptor_model.setInput(face_blob)
            vector = descriptor_model.forward()

            return vector
    return None


# آماده‌سازی داده‌ها
data = {'data': [], 'label': []}

for person in os.listdir('people-images'):
    folder_path = os.path.join('people-images', person)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            vector = extract_face_vector(file_path)
            if vector is not None:
                data['data'].append(vector)
                data['label'].append(person)
        except Exception as e:
            print(f'Error processing {file_path}:', e)

# ذخیره داده‌ها
pickle.dump(data, open('data_face_features.pickle', 'wb'))
print('Feature data saved successfully!')

