import numpy as np
import predict as pd
import cv2

# Start the video feet 
cap = cv2.VideoCapture(0)

# Face global params
face_cascade = cv2.CascadeClassifier('/Users/mehul/opencv-3.0.0/build/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')   

while(True):
    # Get a frame
    ret, frame = cap.read()
    #create a window and resize it 
    cv2.namedWindow('Video Feed', 0)
    cv2.moveWindow('Video Feed', 0, 0)
    cv2.resizeWindow('Video Feed', 720, 400)
    cv2.imshow('Video Feed', frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #display face in gray scale 
    cv2.namedWindow('Face Detected', 0)
    cv2.moveWindow('Face Detected', 720, 0)
    cv2.resizeWindow('Face Detected', 720, 400)
    #cv2.imshow('Face Detected', gray)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
        
    cv2.imshow('Face Detected', gray)        
    

    #cv2.imshow('Gray Scale', gray)
    if cv2.waitKey(1) & 0xFF == ord('p'):
            #this might print a list         
            print 
            print pd.predict(gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            exit(1)        

cap.release()
cv2.destroyAllWindows()
