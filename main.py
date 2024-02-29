import cv2
import numpy as np
import face_recognition
import os

path = 'persons'
images = []
classNames = []
#liste les image dans le dossier persons dans un liste sous le titre personListe
personsList = os.listdir(path)

for cl in personsList:
    #lire tous les image dans le dossier persons
    curPersonn = cv2.imread(f'{path}/{cl}')
    #put all the name of the image in list images
    images.append(curPersonn)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

#cree un fct pour encoding les image 
def findEncodeings(images_list):
    encodeList = []
    for img in images_list:
        #transder l'image from BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #ENCODING THE IMAGE withe the bibleteque face-recogintion
        encode = face_recognition.face_encodings(img_rgb)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodeings(images)
#print(encodeListKnown)

print('Encoding Complete.')

cap = cv2.VideoCapture(1)

while True:
    _, img = cap.read()

    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurentFrame = face_recognition.face_locations(imgS)
    encodeCurentFrame = face_recognition.face_encodings(imgS, faceCurentFrame)
    print(encodeCurentFrame)

    for encodeface, faceLoc in zip(encodeCurentFrame, faceCurentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeface)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeface)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,0,255), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
        else:
            # If face not found in the database, display "Not Found"
            cv2.putText(img, 'Not Found', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Face Recognition', img)
    cv2.waitKey(1)
