import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

People = []

for i in os.listdir(r'C:\Users\asus\Downloads\Computer Vision Project\Dataset'):
    People.append(i)

haar_cascade = cv2.CascadeClassifier('HaarFace_Classifier.xml')

features = []
labels = []

def Create_Dataset():
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

Create_Dataset()
print("DatatSet Created------------------------------------------------")

# Split Dataset into Training and Testing

features = np.array(features, dtype='object')
labels = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=42)
print("DatatSet Splitted into Training and Testing---------------------")

# Create Model and Train Model
print("Training--------------------------------------------------------")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(X_train, y_train)

# Test Model
print("Testing--------------------------------------------------------")
correct_predictions = 0

for i in range(len(y_test)):
    label, confidence = face_recognizer.predict(X_test[i])  # Predict the label for the test image
    if label == y_test[i]:
        correct_predictions += 1

print(f'Avarage Accuracy: {(correct_predictions / len(y_test)) * 100.0}')

face_recognizer.save('Model.yml')
np.save("features.npy", features)
np.save("labels.npy", labels)
