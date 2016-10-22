import numpy as np
import predict as pd
import cv2

# Start the video feet 
cap = cv2.VideoCapture(0)

while(True):
    # Get a frame
    ret, frame = cap.read()

    cv2.imshow('Video Feed', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    if cv2.waitKey(1) & 0xFF == ord('p'):
            #this might print a list         
    	print pd.predict(gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
