import os
import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Path to the folder containing the images
folder_path = 'UTKFace/'

# List all the files in the folder
files = os.listdir(folder_path)


# Create an empty list to hold the images
images = []
ages = []
proportions = []


# Loop through each file in the folder
iterator = 0
for file_name in files:
    if iterator % 5 != 0:
    # Read the image using cv2
        img = cv2.imread(os.path.join(folder_path, file_name))
    # Add the image to the list
        images.append(img)
        age = file_name.split("_")
        ages.append(int(age[0]))
    iterator = iterator + 1



    
    

# Now the images are stored in the `images` list as numpy arrays

# cv2.imshow('First Image', images[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the facial landmark detector


def findDistance(x1, y1, x2, y2):
    a = x2 - x1
    b = y2 - y1
    c2 = a * a + b * b
    return c2 ** .5
counter = 0

LargeDimensionalVector = []
for i in range (0, len(images)):


    # Detect faces in the image
    gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)

    # Iterate through each face detected in the image
    if len(faces) != 1:
        ages.remove(ages[i - counter])
        counter += 1
        counter += 1
    else:
        for face in faces:
            # Draw a rectangle around the face
            #x, y, w, h = face.left(), face.top(), face.width(), face.height()
            #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
        # Detect facial landmarks in the face region
            landmarks = landmark_detector(gray, face)
            data =[]
            for j in range(0, 60):
                data.append(landmarks.part(j).x)
                data.append(landmarks.part(j).y)
            LargeDimensionalVector.append(data)
            forhead = findDistance(landmarks.part(17).x, landmarks.part(17).y, landmarks.part(26).x, landmarks.part(26).y)
            chin = findDistance(landmarks.part(57).x, landmarks.part(57).y, landmarks.part(8).x, landmarks.part(8).y)
            proportion = forhead / chin
            proportions.append(proportion)





    
    # Extract the landmark points for the eyes, nose, mouth, and chin
    # The landmark points are stored as tuples (x, y)
    # left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    # right_eye = (landmarks.part(45).x, landmarks.part(45).y)
    # nose = (landmarks.part(30).x, landmarks.part(30).y)
    # mouth_left = (landmarks.part(48).x, landmarks.part(48).y)
    # mouth_right = (landmarks.part(54).x, landmarks.part(54).y)
    # chin = (landmarks.part(8).x, landmarks.part(8).y)
    
    # # Draw circles at the landmark points
    # cv2.circle(image, left_eye, 2, (0, 0, 255), -1)
    # cv2.circle(image, right_eye, 2, (0, 0, 255), -1)
    # cv2.circle(image, nose, 2, (0, 0, 255), -1)
    # cv2.circle(image, mouth_left, 2, (0, 0, 255), -1)
    # cv2.circle(image, mouth_right, 2, (0, 0, 255), -1)
    # cv2.circle(image, chin, 2, (0, 0, 255), -1)
        # for i in range(1, 68):
        #     cv2.circle(image, (landmarks.part(i).x, landmarks.part(i).y), 2, (0, 0, 255), -1)
        #     if (i == 9):
        #         print(" x is ", landmarks.part(i).x)
        #         print("y is ", landmarks.part(i).y)


# Display the image with the detected face and facial landmarks
# cv2.imshow('Face Detection', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(len(images))
# print(len(ages))
# print(len(proportions))

# plt.plot(ages, proportions, 'o-')

# # Set the x-axis label
# plt.xlabel('Age')

# # Set the y-axis label
# plt.ylabel('Proportions')

# # Show the plot
# plt.show()

b, m = polyfit(ages, proportions, 1)

plt.scatter(ages, proportions, color='blue', label='Data Points')
plt.plot(ages, b + np.array(m) * ages, color='red', label='Line of Best Fit')

equation = f'y = {m:.2f}x + {b:.2f}'
plt.text(25, 0.85, equation, fontsize=10, color='black')

plt.xlabel('Ages')
plt.ylabel('Proportions')
plt.legend()

plt.show()

agesReordered = []
averageProportion = []

for i in range(0, 117):
    agesReordered.append([])

for i in range(len(ages)):
    #if isinstance(ages[i], str):
    #    continue
    agesReordered[ages[i]].append(proportions[i])

def findAverage(arr):
    n = len(arr)
    if n == 0:
        return -1
    sum = 0
    for num in arr:
        sum = sum + num
    return sum / n

for i in range(1, 80):
    averageProportion.append(findAverage(agesReordered[i]))

plt.plot(averageProportion)

plt.xlabel('Actual Age')

# Set the y-axis label
plt.ylabel('Average Proportion')

# Show the plot
plt.show()

print(averageProportion)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(LargeDimensionalVector)
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr_model.fit(X_train_scaled, ages)
filename = 'svr_model.pkl'

with open(filename, 'wb') as file:
    pickle.dump(svr_model, file)
