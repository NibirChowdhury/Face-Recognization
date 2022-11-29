import cv2
import numpy as np

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeDetect=cv2.CascadeClassifier('haarcascade_eye.xml')
cam=cv2.VideoCapture(0)
id=input("Enter you ID:")
sampleNumber=0
while True:
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        sampleNumber=sampleNumber+1
        cv2.imwrite("dataset/User."+str(id)+"."+str(sampleNumber)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=img[y:y+h,x:x+w]
        eyes = eyeDetect.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew,ey+eh), (0,255,0), 2)
    cv2.imshow("Face",img);
    cv2.waitKey(1)
    if (sampleNumber>50):
        break
cam.release()
cv2.destroyAllWindows()
