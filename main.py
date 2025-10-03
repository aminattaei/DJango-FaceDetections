import cv2

net = cv2.dnn.readNetFromCaffe('./models/deploy.prototxt.txt','./models/res10_300x300_ssd_iter_140000_fp16.caffemodel')

image = cv2.imread('./images/big_team.jpg')
h,w = image.shape[:2]

blob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),1,(300,300),(104,177,123),False)

net.setInput(blob)

detections = net.forward()

for i in range(0,detections.shape[2]):
    confidence = detections[0,0,i,2]

    if confidence > 0.5:
        box = detections[0,0,i,3:7] * [w,h,w,h]
        (x1,y1,x2,y2) = box.astype("int")
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,255),2)


cv2.imshow("Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()