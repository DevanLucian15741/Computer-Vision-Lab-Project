import numpy as np
import cv2
import os

def Face_Recognition_Testing():
    People = []
    for i in os.listdir('Dataset'):
        People.append(i)

    haar_cascade = cv2.CascadeClassifier('HaarFace_Classifier.xml')
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    try:
        face_recognizer.read("Model.yml")    
    except cv2.error as e:
        return
    
    face_recognizer.read("Model.yml")
    Directory = input("Input Absolute Path to Predict Image: ")

    img = cv2.imread(Directory)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces_rect =  haar_cascade.detectMultiScale(gray, 1.1, 30)

    for (x,y,w,h) in faces_rect:
        faces = gray[y:y+h,x:x+w]
        faces = cv2.bilateralFilter(faces, 5, 200, 200)
        faces = cv2.GaussianBlur(faces, (11,11), 0, None)
        faces = cv2.cornerHarris(faces, 2, 9, 0.04)
        faces = cv2.resize(faces, (69,69), interpolation = cv2.INTER_AREA)
        label, confidence = face_recognizer.predict(faces)

        if (confidence < 100):
            label = People[label]
            confidence = "  {0}".format(round(100 - confidence))
        else:
            label = "unknown"
            confidence = "  {0}".format(round(100 - confidence))
    
        cv2.putText(img, str(f'{label }:{confidence}'), (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,255), thickness = 2)
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), thickness = 2)

    cv2.imshow("Input Face", img)
    cv2.waitKey(0)

# C:\Users\Asus\Downloads\Computer Vision Project\Dataset\kylian_mbappe\7.jpg
