import cv2

# Start the video feet 
cap = cv2.VideoCapture(0)

while True:

	ret, frame = cap.read()

	cv2.namedWindow('SIFT Keypoints', cv2.WINDOW_NORMAL)
	cv2.moveWindow('SIFT Keypoints', 0, 400)
	cv2.resizeWindow('SIFT Keypoints', 720, 400)

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create()
	kp = sift.detect(gray,None) 
	img=cv2.drawKeypoints(gray,kp, None)

	cv2.imshow('SIFT Keypoints', img)

	if cv2.waitKey(1) & 0xFF == ord('q'):
	            break



cap.release()
cv2.destroyAllWindows()
