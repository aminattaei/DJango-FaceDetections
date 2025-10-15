import cv2
import numpy as np
import pickle

# ---------------------------
# 1. Load trained model & scaler
# ---------------------------
print("📦 در حال بارگذاری مدل و اسکالر ...")
model = pickle.load(open('data_face_features.pickle', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# ---------------------------
# 2. Load face embedding model (OpenFace)
# ---------------------------
print("🧠 در حال بارگذاری مدل استخراج ویژگی چهره ...")
face_description_model = './models/openface.nn4.small2.v1.t7'
embedder = cv2.dnn.readNetFromTorch(face_description_model)

# ---------------------------
# 3. Read new face image
# ---------------------------
image_path = 'test_image.jpg'  # مسیر تصویر جدید
image = cv2.imread(image_path)

if image is None:
    print("❌ تصویر پیدا نشد! مسیر فایل رو چک کن.")
    exit()

(h, w) = image.shape[:2]

# ---------------------------
# 4. Detect face (optional)
# ---------------------------
# اگر از قبل چهره رو برش دادی، این مرحله رو لازم نداری.
# در غیر این صورت می‌تونی از مدل DNN یا haarcascade استفاده کنی برای شناسایی چهره‌ها.

# ---------------------------
# 5. Create blob & extract features
# ---------------------------
face_blob = cv2.dnn.blobFromImage(cv2.resize(image, (96, 96)), 1.0 / 255,
                                  (96, 96), (0, 0, 0), swapRB=True, crop=False)

embedder.setInput(face_blob)
vec = embedder.forward()  # خروجی 128 ویژگی چهره

# ---------------------------
# 6. Normalize & Predict
# ---------------------------
vec_scaled = scaler.transform(vec.reshape(1, -1))
prediction = model.predict(vec_scaled)
prob = model.predict_proba(vec_scaled)

confidence = np.max(prob) * 100
person = prediction[0]

# ---------------------------
# 7. Print Result
# ---------------------------
print(f"✅ نتیجه تشخیص: {person}")
print(f"📊 اطمینان مدل: {confidence:.2f}%")

# ---------------------------
# 8. Optional — نمایش تصویر با نتیجه
# ---------------------------
text = f"{person} ({confidence:.1f}%)"
cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
cv2.imshow("Prediction", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
