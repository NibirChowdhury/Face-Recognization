import cv2
import numpy as np 
import os

fname = "recognizer/trainingData.yml"
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeDetect=cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(fname)
id=0
while (True):
  ret, img = cap.read()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    roi_gray=gray[y:y+h,x:x+w]
    roi_color=img[y:y+h,x:x+w]
    eyes = eyeDetect.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
      cv2.rectangle(roi_color, (ex, ey), (ex+ew,ey+eh),(0,0,255), 2)
      id,conf = recognizer.predict(gray[y:y+h,x:x+w])
    if (id==1):
      cv2.putText(img,'SIMUL', (x+2,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (176,224,230),3)
    elif(id==2):
      cv2.putText(img,'PRAPTI',(x+2,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (176,224,230),3)
    elif(id==3):
      cv2.putText(img,'tusar',(x+2,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (176,224,230),3)
    elif(id==4):
      cv2.putText(img,'Pranto',(x+2,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (176,224,230),3)
    
    else:
      cv2.putText(img, 'No Match', (x+2,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (176,224,230),3)
  cv2.imshow('Face Recognizer',img)
  k = cv2.waitKey(30) & 0xff
  if k == 27:
    break
cap.release()
cv2.destroyAllWindows()
