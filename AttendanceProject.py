# Step 1: Importing the required packages
# packages or dependencies added: cmake, dlib, face-recognition, numpy, opencv

#importing packages
import cv2
import numpy as np
import face_recognition
import os                                          #this helps us find images folder and encode the images in them
from datetime import datetime

# Step 2: Importing images
path = 'ImagesAttendance'
images = []                                        #list of all images that we import
classNames = []                                    #to take name of person from images
myList=os.listdir(path)                            #grabbing list of images from folder

for cl in myList:                                  #use the images and import the images one by one
    curImg = cv2.imread(f'{path}/{cl}')            #imread is a opencv function
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])     #to remove the '.jpg' from name of image while displaying
print(classNames)

# Step 3: Finding the encodings for each image
def findEncodings(images):
    encodeList = []                                #empty list to have all the encodings at the end
    for img in images:                             #looping through all the images
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)           #convertimg the image into RGB
        encode = face_recognition.face_encodings(img)[0]     #getting the encodings
        encodeList.append(encode)                            #adding the encodings to the list
    return encodeList

# Step 7: Marking the attendance (name, time, date) in the Excel file
def markAttendance(name):                                    #marking the attendance
    with open('Attendance.csv','r+') as f:                   #opening attendance file for read and write
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:                              #to avoid repition of attendance
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:                              #if the attendance of the person is not done already then taking the attendance
            now = datetime.now()
            tStr = now.strftime('%H:%M:%S')                  #time of taking attendance
            dStr = now.strftime('%d/%m/%Y')                  #date
            f.writelines(f'\n{name},{tStr},{dStr}')          #output

encodeListKnown = findEncodings(images)
print(len(encodeListKnown))
print('Encoding Complete')

#Step 4: Initializing the webcam and obtaining the face locations and encodings
cap = cv2.VideoCapture(0)

while True:                                                 #loop to get each frame one by one
    success, img = cap.read()                               #this will give us our image
    imgS = cv2.resize(img, (0,0), None, 0.25,0.25)          #reducing the size of image to make the process faster(reduced by 1/4th)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
                                                            #step: finding the encodings of our webcam
    facesCurFrame = face_recognition.face_locations(imgS)   #finding locations of our faces and then sending it to encoding functions
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

# Step 5: Finding the match between our encodings and the current webcam encodings
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        #it will grab face loc from facesCurFrame list and then it will grab the encoding of encode face from the encodesCurFrame
        #using zip as we want them in the same loop
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)     #matching the encodings of image with images in encodeListKnown
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)     #finding the face distance
        matchIndex = np.argmin(faceDis)                                          #give index of the image that matches the most with the webcam image

# Step 6: To show a bounding box and name of the person while recognizing the face
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)                                                          #printing the recognized face
            y1,x2,y2,x1 = faceLoc                                                #code for rectangle around webcam image
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4                              #as we had reduced the image size above by one-fourth hence 4 multiplied here
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)                                                 #after match found marking the attendance

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)



