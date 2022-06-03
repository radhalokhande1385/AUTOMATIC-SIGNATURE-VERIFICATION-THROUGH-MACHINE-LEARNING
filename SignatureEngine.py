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
import numpy as np

import cv2

from numpy import asarray
import os

import time

import datetime
import mysql.connector
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
path="dataset/train"
dir_list = os.listdir(path)
signature_dict={}
i=0
for x in dir_list:
    foldername=x
    signature_dict[i]=foldername
    i=i+1
while True:
    # Find haar cascade to draw bounding box around face
    print("Visited")
    time.sleep(3)
    imagedirname="D://Signature"
    filelist=[]
    for path in os.listdir(imagedirname):
        imagepath = os.path.join(imagedirname, path) 
        filelist.append(imagepath)
    
    totalfiles=len(filelist) 
    if(totalfiles!=0):
        signaturepath=filelist[0]
        print("Image found at path ",signaturepath)
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
        os.remove(signaturepath)
        statustoinsert=""
        if(status=='F'):
            statustoinsert="Forge"
        if(status=='R'):
            statustoinsert="Real"
            
        mydb = mysql.connector.connect( host="localhost", user="root",  passwd="root",  database="dpcoesignatureverification")

        mycursor = mydb.cursor()
      #pat_id, mood, time #student_id, signature_status
        sql = "INSERT INTO resultinfo (student_id, signature_status) VALUES (%s, %s)"
        val = (withstr, statustoinsert)

        mycursor.execute(sql, val)

        mydb.commit()

        print(mycursor.rowcount, "record inserted.")    
            
        
#         fstr=os.path.basename(image_path)
#         x = fstr.split(".")
#         patientnumber=x[0]
        
#         if os.path.isfile(image_path):
#             img = Image.open(image_path)
#             frame = asarray(img)
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             result_string="none"
#             cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray, (150, 150)), -1), 0)
#             prediction = model.predict(cropped_img)
#             maxindex = int(np.argmax(prediction))
#             result_string=disease_dict[maxindex]
           
            
#         os.remove(image_path)
#         print("result_string is ",result_string)
#         print("Patient NUmber is ",patientnumber)
        
#         status_string=""
#         if(result_string=="NORMAL"):
#             status_string="NEGATIVE"
#         if(result_string=="PNEUMONIA"):
#                 status_string="POSITIVE"    
            
#         print("Inserting value is "+status_string)
#         mydb = mysql.connector.connect( host="localhost", user="root",  passwd="root",  database="covid19predictionsystem")

#         mycursor = mydb.cursor()
#       #pat_id, mood, time
      
     
    
#         sql="UPDATE patientdetailsfortest SET covid_status='"+status_string+"' where sr_no='"+patientnumber+"' ";
      
#         print(sql)
       

#         mycursor.execute(sql)
# #
#         mydb.commit()

#         print(mycursor.rowcount, "record Updated.")
       
        
        
       
