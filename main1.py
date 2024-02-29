# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 19:36:32 2023

@author: R I B
"""

import cv2
import numpy as np
import face_recognition
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

path = 'persons'
images = []
classNames = []
labels = []

# List the images in the 'persons' folder
personsList = os.listdir(path)

for cl in personsList:
    # Read all images in the 'persons' folder
    curPersonn = cv2.imread(f'{path}/{cl}')
    # Put all the names of the images in the 'images' list
    images.append(curPersonn)
    classNames.append(os.path.splitext(cl)[0])
    labels.append(classNames.index(os.path.splitext(cl)[0]))

print(classNames)

# Create a function to encode the images
def findEncodings(images_list):
    encodeList = []
    for img in images_list:
        # Transfer the image from BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Encoding the image using the face-recognition library
        encode = face_recognition.face_encodings(img_rgb)[0]
        encodeList.append(encode)
    return encodeList

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Encode the training set
encodeListKnown = findEncodings(X_train)

# Train the k-NN model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(encodeListKnown, y_train)

print('Training Complete.')

cap = cv2.VideoCapture(1)

while True:
    _, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurrentFrame = face_recognition.face_locations(imgS)
    encodeCurrentFrame = face_recognition.face_encodings(imgS, faceCurrentFrame)
    print(encodeCurrentFrame)

   # Predict labels for the faces in the current frame
    encodeCurrentFrame_np = np.array(encodeCurrentFrame).reshape(-1, len(encodeCurrentFrame[0]))
    labels_predicted = knn_model.predict(encodeCurrentFrame_np)
    
    # Display the histogram of predicted labels
    plt.figure(figsize=(8, 6))
    plt.hist(labels_predicted, bins=len(classNames), align='left', rwidth=0.8, color='blue', alpha=0.7)
    plt.xticks(range(len(classNames)), classNames, rotation=45)
    plt.xlabel('Classes')
    plt.ylabel('Number of Faces')
    plt.title('Number of Faces in Each Class')
    plt.tight_layout()
    plt.show()

    for encodeFace, faceLoc in zip(encodeCurrentFrame, faceCurrentFrame):
    # Check if face encodings are found
     if encodeFace.any():
        # Use the k-NN model to predict the label
        label = knn_model.predict([encodeFace])[0]
        name = classNames[label].upper()
        print(name)

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Face Recognition', img)
    cv2.waitKey(1)
