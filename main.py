import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    cv2.imshow('Video Feed', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
    if cv2.waitKey(1) & 0xFF == ord('p'):
    	predict(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
