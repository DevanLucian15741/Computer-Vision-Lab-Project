from Model_Train import *;
from Face_Reconition import *;

def Print_Menu():
    print("Choose a number between [1-3]")
    print("1-> Train and Test Model")
    print("2-> Predict")
    print("3-> Exit")

choice = 1

while(int(choice)<3):
    Print_Menu()
    choice = input("Enter Menu option: ")
    if(int(choice)==1):
        FaceRecognition_Training()
    elif(int(choice)==2):
        Face_Recognition_Testing()
    else:
        print("Thank You, Goodbye!")
    
    