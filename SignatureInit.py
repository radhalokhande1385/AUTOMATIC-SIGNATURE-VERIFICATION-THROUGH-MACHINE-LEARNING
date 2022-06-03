import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from keras.models import load_model



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def initTesting(signaturepath):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(150,150,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(24, activation='softmax'))
    model.load_weights('signature_trained_model.h5')
    
    
        # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)
    
        # dictionary which assigns each label an emotion (alphabetical order)
   # signature_dict = {0: "1_F", 1: "1_R",2: "2_F", 3: "2_R"}
    path="dataset/train"
    dir_list = os.listdir(path)
    signature_dict={}
    i=0
    for x in dir_list:
        foldername=x
        signature_dict[i]=foldername
        i=i+1

   # print(signature_dict)  
 
    img = cv2.imread(signaturepath)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dim = (150, 150)
    resized = cv2.resize(gray_image, dim, interpolation = cv2.INTER_AREA)
    sig_img = np.expand_dims(np.expand_dims(cv2.resize(resized, (150, 150)), -1), 0)
    prediction = model.predict(sig_img)
    maxindex = int(np.argmax(prediction))
    print("Matched index is ",maxindex)
    signature_name=signature_dict[maxindex]
    print("matched Signature is ",signature_name)
    st=signature_name.split("_")
    withstr=st[0]
    status=st[1]
    if(status=="R"):
        window_name = 'SIGNATURE MATCHED WITH '+ withstr
        ims=cv2.resize(resized,(500,500))
        cv2.imshow(window_name, ims)
        
    if(status=="F"):
        window_name = 'SIGNATURE FORGED FOR '+withstr
        ims=cv2.resize(resized,(500,500))
        cv2.imshow(window_name, ims)
        
    
    
    
    
if __name__ == '__main__':
    initTesting()        