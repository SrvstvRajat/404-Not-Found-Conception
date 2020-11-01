import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

camera_feedback = True
path = 'Training'
images = []
classNames = []
myList = os.listdir(path)
# print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        try:
            enc = encode[0]
        except:

            print("I am ready\n", end=" ")
            input("Hit Enter To continue")
        encodeList.append(enc)
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


encodeListKnown = findEncodings(images)
print(len(encodeListKnown))
cap = cv2.VideoCapture(0)
# print("I am ready ", end=" ")
# input("Hit Enter To continue")
while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurrFrame = face_recognition.face_locations(imgS)
    encodesCurrFrame = face_recognition.face_encodings(imgS, facesCurrFrame)

    ret, frame = cap.read()

    if camera_feedback:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    for encodeFaces, faceLoc in zip(encodesCurrFrame, facesCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFaces)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFaces)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if ((float(f'{round(faceDis[matchIndex])}')) < 0.55):
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f'{matches[matchIndex]} {round(faceDis[matchIndex], 2)}', (50, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)
        else:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(img, f'{print("Please Add Your Image")}', (50, 50),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            #print("Please Add your Image")

    cv2.imshow('Webcam', img)
    cv2.waitKey(1) & 0xFF == ord('q')

