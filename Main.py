import os
import cv2
import numpy as np

People = []

for i in os.listdir(r'C:\Users\asus\Downloads\Computer Vision Project\Dataset'):
    People.append(i)

haar_cascade = cv2.CascadeClassifier('HaarFace_Classifier.xml')

features = []
labels = []

def create_trainset():
    for person in People:
        path = os.path.join(r'C:\Users\asus\Downloads\Computer Vision Project\Dataset', person)
        label = People.index(person)

        for image in os.listdir(path):
            img_path = os.path.join(path,image)
            img_array = cv2.imread(img_path)
            gray_IMG = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

            faces_rect =  haar_cascade.detectMultiScale(gray_IMG, scaleFactor = 1.1, minNeighbors = 4)
            
            for (x,y,w,h) in faces_rect:
                faces = gray_IMG[y:y+h,x:x+w]
                features.append(faces)
                labels.append(label)

create_trainset()
print("Trains Set Creation Done---------------------")

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.train(features, labels)

face_recognizer.save('Model.yml')
np.save("features.npy", features)
np.save("labels.npy", labels)
