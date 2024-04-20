import cv2
import numpy as np
import face_recognition
cap= cv2.VideoCapture(0)
cap.set(3,720)
cap.set(4,1080)
success=True
while success:
    success, frame = cap.read()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
    image = frame
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow('Detected Faces', image)
    key=cv2.waitKey(10)
