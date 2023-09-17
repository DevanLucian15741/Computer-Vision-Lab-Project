import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def FaceRecognition_Training():
   
    haar_cascade = cv2.CascadeClassifier('HaarFace_Classifier.xml')

    # Data Structures
    image_list = []
    Labels_List = []

    # Acessing Dataset
    train_path = 'Dataset'
    train_dir_List = os.listdir(train_path)

    for i, Folder in enumerate(train_dir_List):
        Data_path_Folder = train_path+'/'+Folder
        for Image in os.listdir(Data_path_Folder)[:-1]:
            Data_Path_Images = Data_path_Folder+'/'+Image
            img_gray = cv2.imread(Data_Path_Images, 0)
            Extract_Faces = haar_cascade.detectMultiScale(img_gray, scaleFactor = 1.1, minNeighbors = 30)# Set MinNeighbors to 30 to avoid haar cascade detecting more than one faces

            if len(Extract_Faces) < 1:
                 continue
            for (x,y,w,h) in Extract_Faces:
                    faces = img_gray[y:y+h,x:x+w]
                    # faces = cv2.resize(faces, (100, 100 ), interpolation = cv2.INTER_AREA)
                    # Apply bilateral Blur and GuassianBlur as a part of Data Augmentation To Increase Accuracy, reduce Noise
                    faces = cv2.bilateralFilter(faces, 5, 200, 200)
                    faces = cv2.GaussianBlur(faces, (11,11), 0, None)

                    # faces = cv2.blur(faces, (11,11))
                    # faces = cv2.medianBlur(faces, 11)
                    # faces = cv2.fastNlMeansDenoising(faces, 3, None, 7, 21)
                    
                    image_list.append(faces)
                    Labels_List.append(i)
    
    print("DatatSet Created...")

    image_list = np.array(image_list, dtype='object')
    Labels_List = np.array(Labels_List)
    
    #Split Dataset into Training and Testing using Sklearn
    X_train, X_test, y_train, y_test = train_test_split(image_list, Labels_List, test_size=0.25, random_state=97)
    X_train_resized = [cv2.resize(image, (50, 50), interpolation = cv2.INTER_AREA) for image in X_train]
    X_test_resized = [cv2.resize(image, (50, 50), interpolation = cv2.INTER_AREA) for image in X_test]

    print("DatatSet Splitted into Training and Testing...")

    # Create Model and Train Model
    print("Training...")
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(X_train_resized, y_train)

    # Test Model
    print("Testing...")
    
    num_imgs = len(X_test_resized)
    num_correct = 0
    for img, label in zip(X_test_resized,y_test):
        Predlabel,_ = face_recognizer.predict(img)  # Predict the label for the test image
        if label == Predlabel:
            num_correct += 1

    print('Validation: ', num_correct, 'correct out of', num_imgs)

    print(f'Avarage Accuracy: {(num_correct / len(X_test_resized)) * 100.0}')

    face_recognizer.save('Model.yml')
    np.save("features.npy", image_list)
    np.save("labels.npy", Labels_List)

# random_state=np.random.randint(100)