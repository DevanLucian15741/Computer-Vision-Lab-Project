from Model_Train import *;
from Face_Reconition import *;

def Print_Menu():
    print("Choose a number between [1-3]")
    print("1-> Train and Test Model")
    print("2-> Predict")
    print("3-> Exit")

MenuOption = [lambda: FaceRecognition_Training(),
              lambda: Face_Recognition_Testing(),
              lambda: print("Thank You, Goodbye!")]

choice = -1
while(int(choice)<3):
    try:
        Print_Menu()
        choice = int(input("Enter Menu option: "))
        MenuOption[int(choice-1)]()
    except ValueError:
        print("Invalid input. Please enter an integer.")
    
    