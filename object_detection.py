# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 16:01:52 2019

@author: vrhar
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
cap = cv2.VideoCapture(1)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
i=0
frames=[]
counter=0
contours_final_array=[]
while(cap.isOpened() and i<10):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frames.append(frame)
    imgg= np.copy(frame)
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    img_gray = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)
    light = (0, 100, 20)
    dark = (30, 255, 255)
    mask = cv2.inRange(img_hsv, light, dark)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    kernel = np.ones((30,30),np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    ret,thresh = cv2.threshold(closing,127,255,0)
    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours_final=[]
    for i in range(len(contours)):
        if (cv2.contourArea(contours[i])>10000):
            contours_final.append(contours[i])
        print(cv2.contourArea(contours[i]))
    print(len(contours_final))
    img = cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    last_index=len(frames)-1
    contours_final_array.append(contours_final)
   
    if(len(contours_final_array)>10):
        for i in range(10):
            contours_final_array[i]=contours_final_array[i+1]
        del contours_final_array[10]
        
    if(len(frames)>10):
        for i in range(10):
            frames[i]=frames[i+1]
        del frames[10]
        
    if(len(contours_final_array)>9):    
        if (len(contours_final_array[9])!=len(contours_final_array[0])):
            print('send picture')
            
    plt.subplot(1, 2, 1)
    plt.imshow(img_hsv)
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.show()
    
cap.release()
cv2.destroyAllWindows()
