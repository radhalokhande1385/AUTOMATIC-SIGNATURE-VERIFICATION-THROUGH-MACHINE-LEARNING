import os
import cv2




def walkdir(dirname):
    count=0
    currentdirpath=""
    for cur, _dirs, files in os.walk(dirname):
        pref = ''
        head, tail = os.path.split(cur)
        while head:
           # pref += '---'
            head, _tail = os.path.split(head)
            currentdirpath=dirname+pref+"//"+tail
            print(currentdirpath)
        for f in files:
            finalpath=currentdirpath+"//"+f
            print("Final path : ",finalpath)
            img = cv2.imread(finalpath)
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dim = (150, 150)
            resized = cv2.resize(gray_image, dim, interpolation = cv2.INTER_AREA)
            cv2.imwrite(finalpath,resized)
            count=count+1
    print("Count is ",count)        
            
            
dirname="dataset//train"


walkdir(dirname)
        
        
        
        
        
