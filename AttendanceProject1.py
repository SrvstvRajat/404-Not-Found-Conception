import cv2
import numpy as np
import face_recognition
import os

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
            print("[ERROR] Can't find face in image")
        encodeList.append(enc)
    return encodeList


encodeListKnown = findEncodings(images)
print(len(encodeListKnown))
cap = cv2.VideoCapture(0)
print("I am ready ", end=" ")
input("Hit Enter To continue")
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
        if camera_feedback:
            cv2.rectangle(img, (faceLoc[0], faceLoc[1]),
                          (faceLoc[2], faceLoc[3]), (0, 255, 0), 2)
            cv2.imshow('frame', frame)
        try:
            matches = face_recognition.compare_faces(
                encodeListKnown, encodeFaces)
            faceDis = face_recognition.face_distance(
                encodeListKnown, encodeFaces)

            print(faceLoc)
            print("Name" + " "*21 + "Chances")
            for x in range(len(classNames)):
                print("{:25s} {:20f}".format(classNames[x], faceDis[x]*100))
        except ValueError as e:
            print("[ERROR]" + str(e))
            print("Face: ")
            print(encodeFaces)
            print("Location: ")
            print(faceLoc)


#faceLoc = face_recognition.face_locations(imgElon)[0]
#encodeElon = face_recognition.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

#faceLocTest = face_recognition.face_locations(imgTest)[0]
#encodeTest = face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

#results = face_recognition.compare_faces([encodeElon],encodeTest)
#faceDis = face_recognition.face_distance([encodeElon],encodeTest)
