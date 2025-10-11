import numpy as np
import pandas as pd
import pickle
import cv2


data = pickle.load(open('data_face_features.pickle',mode='rb'))

x = np.array(data['data'])
y = np.array(data['label'])




print(x.shape)
print(y.shape)