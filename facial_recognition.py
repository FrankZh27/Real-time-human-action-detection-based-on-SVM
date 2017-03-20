import cv2
import numpy
#face_cascade = cv2.CascadeClassifier('D:/programfile/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier('D:/programfile/opencv/opencv/sources/data/haarcascades/haarcascade_fullbody.xml')

face_cascade = cv2.HOGDescriptor()
face_cascade.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#eye_cascade = cv2.CascadeClassifier('D:/programfile/opencv/opencv/sources/data/haarcascades/haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('person15_walking_d1_uncomp.avi')
ret, img = cap.read()
while ret:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#    faces = face_cascade.detectMultiScale(gray,1.3, 5)
#    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, flags=cv2.CASCADE_SCALE_IMAGE)
    faces,w = face_cascade.detectMultiScale(gray)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w, y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]
        '''
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        '''
    
    cv2.imshow('img',img)
    k = cv2.waitKey(30)& 0xff
    if k == 27:
        break
    ret, img = cap.read()

cap.release()
cv2.destroyAllWindows()
