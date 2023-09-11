import numpy as np
import cv2
import os

People = []

for i in os.listdir(r'C:\Users\asus\Downloads\Computer Vision Project\Dataset'):
    People.append(i)

haar_cascade = cv2.CascadeClassifier('HaarFace_Classifier.xml')
# features = np.load('features.npy')
# labels = np.load('labels.npy')

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("Model.yml")

img = cv2.imread(r'C:\Users\asus\Downloads\Computer Vision Project\Dataset\erling_haaland\5.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('Person',gray)
faces_rect =  haar_cascade.detectMultiScale(gray, 1.1, 4)
for (x,y,w,h) in faces_rect:
    faces = gray[y:y+h,x:x+w]
    label, confidence = face_recognizer.predict(faces)

    print(f'Label = {People[label]} with a confidence of {confidence}')
    cv2.putText(img, str(People[label]) ,(20,20),cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,255), thickness = 2)
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), thickness = 2)

cv2.imshow("Input Face", img)
cv2.waitKey(0)