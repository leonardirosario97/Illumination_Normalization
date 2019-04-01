#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import os
from skimage import feature
import matplotlib.pyplot as plt
import time

def show_image(image,label='image'):
    cv2.imshow(label,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def adjust_gamma_1(image, gamma=0.2, use_threshold=False, threshold=0):
    im_copy = image.copy()
    for i in range(im_copy.shape[0]):
        for j in range(im_copy.shape[1]):
            if((use_threshold) & (im_copy[i][j]>threshold)):
                im_copy[i][j] = threshold**gamma
            else:
                im_copy[i][j] = image[i][j]**gamma
    return im_copy

def doG(image, sigma0=1, sigma1=2):
    g1 = cv2.GaussianBlur(image,(0,0),sigma0,borderType=cv2.BORDER_REPLICATE)
    g2 = cv2.GaussianBlur(image,(0,0),sigma1,borderType=cv2.BORDER_REPLICATE)
    doG = g1 - g2
    return doG

def contrast_equalization(image,a=0.1, t=10):    
    im_copy = image.copy()
    d1 = adjust_gamma_1(abs(im_copy), gamma=a).mean()**(1/a)
    
    for i in range(im_copy.shape[0]):
        for j in range(im_copy.shape[1]):
            im_copy[i][j] = im_copy[i][j] / d1

    d2 = adjust_gamma_1(abs(im_copy),gamma=a,use_threshold=True, threshold=10).mean()**(1/a)
    for i in range(im_copy.shape[0]):
        for j in range(im_copy.shape[1]):
            im_copy[i][j] = im_copy[i][j]/d2
    
    for i in range(im_copy.shape[0]):
        for j in range(im_copy.shape[1]):
            im_copy[i][j] = t*np.tanh(im_copy[i][j]/t)
            
    return im_copy

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float64)

    #gamma correction
    img_gamma = adjust_gamma_1(img,gamma=0.2)

    #DOG
    img_dog = doG(img_gamma,sigma0=1,sigma1=2)
    
    #Contrast Equalization
    img_contrast_equalization = contrast_equalization(img_dog,a=0.1,t=10)
    return cv2.normalize(img_contrast_equalization, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)

def chi2_distance(histA, histB, eps = 1e-10):
    #d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])
    d = np.sum([((a - b) ** 2) / (a + b) for (a, b) in zip(histA, histB)])
    return d

def distance(h1,h2):
    return np.abs(h1-h2).sum()
    
def resize_same(im1,im2):
    im1 = cv2.resize(im1,(150, 150), interpolation = cv2.INTER_CUBIC)
    im2 = cv2.resize(im2,(150, 150), interpolation = cv2.INTER_CUBIC)
    return im1,im2

def percentage_value(obj):
    lista = []
    for i in zip(set(obj)):
        lista.append(len(obj[i==obj])/len(obj))
    return np.array(lista)

def lbp(im):
    lbp = feature.local_binary_pattern(im,P=8,R=1,method='nri_uniform').reshape(1,-1)[0]
    return lbp

def robustness():
    dataset_path = "./dataset/"
    distance_with_preproc = distance_without_preproc = 0
    iteration=0
    for file in os.listdir(dataset_path):
            if(int(file[2])!=2):
                iteration+=1
                file_path_1 = os.path.join(dataset_path,file)
                file_path_2 = os.path.join(dataset_path,file.replace("1","2",1))
                
                im1 = cv2.imread(file_path_1)
                im2 = cv2.imread(file_path_2)
                im1, im2 = resize_same(im1,im2)
                
                im1_preproc = preprocessing(im1)
                im2_preproc = preprocessing(im2)
                
                im1_lbp = lbp(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY))
                im2_lbp = lbp(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY))
                im1_preproc_lbp = lbp(im1_preproc)
                im2_preproc_lbp = lbp(im2_preproc)
                
                x1 = percentage_value(im1_lbp)
                x2 = percentage_value(im2_lbp)
                y1 = percentage_value(im1_preproc_lbp)
                y2 = percentage_value(im2_preproc_lbp)
                
                distance_without_preproc+=chi2_distance(x1,x2)
                distance_with_preproc+=chi2_distance(y1,y2)

    distance_with_preproc/=iteration
    distance_without_preproc/=iteration
    print("mean chi_distance without preproc: ",distance_without_preproc)
    print("mean chi_distance with preproc: ",distance_with_preproc)

def videoconversion(pathVideo,pathSave='./video_edit.mp4',framerate=30):
  
    vidcap = cv2.VideoCapture(pathVideo)
    success,frame = vidcap.read()
    frame = cv2.resize(frame,(640,360), interpolation = cv2.INTER_CUBIC)

    height_video , width_video , _ =  frame.shape
    video_reg = cv2.VideoWriter(pathSave,cv2.VideoWriter_fourcc(*'mp4v'), framerate ,(width_video,height_video))
    
    while success:
        start_time = time.time()  
        frame = cv2.resize(frame,(640,360), interpolation = cv2.INTER_CUBIC)
        frame=preprocessing(frame)
        
        cv2.imshow('frame',frame)
        print("fps "+ str("%.2f" %  (1.0 / (time.time() - start_time))))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        video_reg.write(frame)
        success,frame = vidcap.read()
    
    video_reg.release()
    vidcap.release()
    cv2.destroyAllWindows()

def robustness():
    dataset_path = "./dataset/"
    distance_with_preproc = distance_without_preproc = 0
    iteration=0
    for file in os.listdir(dataset_path):
            if(int(file[2])!=2):
                iteration+=1
                file_path_1 = os.path.join(dataset_path,file)
                file_path_2 = os.path.join(dataset_path,file.replace("1","2",1))
                
                im1 = cv2.imread(file_path_1)
                im2 = cv2.imread(file_path_2)
                im1, im2 = resize_same(im1,im2)
                
                im1_preproc = preprocessing(im1)
                im2_preproc = preprocessing(im2)
                
                im1_lbp = lbp(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY))
                im2_lbp = lbp(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY))
                im1_preproc_lbp = lbp(im1_preproc)
                im2_preproc_lbp = lbp(im2_preproc)
                
                x1 = percentage_value(im1_lbp)
                x2 = percentage_value(im2_lbp)
                y1 = percentage_value(im1_preproc_lbp)
                y2 = percentage_value(im2_preproc_lbp)
                
                distance_without_preproc+=chi2_distance(x1,x2)
                distance_with_preproc+=chi2_distance(y1,y2)

    distance_with_preproc/=iteration
    distance_without_preproc/=iteration
    print("mean chi_distance without preproc: ",distance_without_preproc)
    print("mean chi_distance with preproc: ",distance_with_preproc)


im1 = cv2.imread('./dataset/A_1.png')
im2 = cv2.imread('./dataset/A_2.png')
im1, im2 = resize_same(im1,im2)

im1_preproc = preprocessing(im1)
im2_preproc = preprocessing(im2)

cv2.imshow('1',im1)
cv2.imshow('2',im1_preproc)
cv2.imshow('3',im2)
cv2.imshow('4',im2_preproc)
cv2.waitKey(0)
cv2.destroyAllWindows()

im1_lbp = lbp(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY))
im2_lbp = lbp(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY))

im1_preproc_lbp = lbp(im1_preproc)
im2_preproc_lbp = lbp(im2_preproc)

plt.figure(figsize=(16,10))
plt.subplot(331)
plt.hist(im1_lbp,bins=59,density=True)
plt.subplot(332)
plt.hist(im2_lbp,alpha=0.5,bins=59,density=True)
plt.subplot(333)
plt.hist(im1_lbp,bins=59,density=True)
plt.hist(im2_lbp,alpha=0.5,bins=59,density=True)

plt.subplot(334)
plt.hist(im1_preproc_lbp,alpha=1,bins=59,density=True)
plt.subplot(335)
plt.hist (im2_preproc_lbp,alpha=0.4,bins=59,density=True)
plt.subplot(336)
plt.hist(im1_preproc_lbp,bins=59,density=True)
plt.hist(im2_preproc_lbp,alpha=0.5,bins=59,density=True)
plt.show()

x1 = percentage_value(im1_lbp)
x2 = percentage_value(im2_lbp)
y1 = percentage_value(im1_preproc_lbp)
y2 = percentage_value(im2_preproc_lbp)

print("Chi2_distance im without preproc: ",chi2_distance(x1,x2))
print("Chi2_distance im with preproc: ", chi2_distance(y1,y2))


robustness()
#videoconversion(pathVideo="./Video/ball_1.mp4",pathSave='./Video/ball_1_edit_without_optimization.mp4')